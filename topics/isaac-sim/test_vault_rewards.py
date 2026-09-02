"""Check the `vault` reward math against a hand-computed trench. No Kit, no GPU, no sim.

    source env.sh && "$ISAACSIM_PYTHON_EXE" test_vault_rewards.py

Runs in about two seconds, and that is the entire reason it exists. Everything else in
this repo that can tell you whether a reward term is right costs a Kit boot and a few
hundred training iterations, so the feedback loop on "is `gap_flight` actually zero when
the robot is walking?" was otherwise twenty minutes. The two terms in
`spark_envs._vault_profile` are pure functions of a height-scan tensor, a velocity and a
contact time, so they can be fed a fake trench and checked against arithmetic anyone can
do on paper.

It needs the Isaac python only for torch and for importing `spark_envs`, which pulls in
`isaaclab`'s config dataclasses. Nothing here boots a `SimulationApp`.

The grid below is the real one: `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])` from
`velocity_env_cfg.py:112`, built the same way `patterns.grid_pattern` builds it, so the
187 rays and their ordering match what the sensor actually produces.
"""

import sys
import types

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import spark_envs as se  # noqa: E402

R_X = torch.arange(-0.8, 0.8 + 1e-9, 0.1)           # 17 columns, +x is robot-forward
R_Y = torch.arange(-0.5, 0.5 + 1e-9, 0.1)           # 11 rows
GX, GY = torch.meshgrid(R_X, R_Y, indexing="xy")    # ordering="xy", as the cfg defaults
STARTS = torch.stack([GX.flatten(), GY.flatten(), torch.zeros(GX.numel())], dim=-1)

# The tuned constants from _vault_profile, at Go2 scale (scale == 1.0).
AHEAD, UNDER, DEPTH, Y_HALF = (0.10, 0.70), (-0.40, 0.40), 0.30, 0.30


class Proxy:
    """Stands in for isaaclab's ProxyArray, which wraps a warp array and exposes .torch."""

    def __init__(self, t):
        self.torch = t


def make_env(ground_z, vel=(1.0, 0.0, 0.0), air=(0.1, 0.1, 0.1, 0.1)):
    """One env on a patch whose origin is z=0.

    `ground_z` maps a ray's local x to the world z it hits, which is how a trench, a rail
    or a missed ray get described below. `vel` is (forward in BASE frame, y, up in WORLD
    frame): gap_takeoff reads those from two different frames on purpose, so the mock has
    to keep them apart or the wheelie test below would pass for the wrong reason.
    """
    hits = torch.zeros(1, STARTS.shape[0], 3)
    hits[0, :, 0] = STARTS[:, 0]
    hits[0, :, 1] = STARTS[:, 1]
    hits[0, :, 2] = torch.tensor([ground_z(float(x)) for x in STARTS[:, 0]])

    scanner = types.SimpleNamespace(
        ray_starts=Proxy(STARTS.unsqueeze(0)),
        data=types.SimpleNamespace(ray_hits_w=Proxy(hits)),
    )
    contact = types.SimpleNamespace(
        data=types.SimpleNamespace(current_air_time=Proxy(torch.tensor([list(air)])))
    )
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_lin_vel_b=Proxy(torch.tensor([[vel[0], vel[1], 0.0]])),
        root_lin_vel_w=Proxy(torch.tensor([[0.0, vel[1], vel[2]]])),
    ))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)
        sensors = {"height_scanner": scanner, "contact_forces": contact}

    return types.SimpleNamespace(scene=Scene(robot=robot))


SCAN = types.SimpleNamespace(name="height_scanner")
CONTACT = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
ROBOT = types.SimpleNamespace(name="robot")

# A 0.26 m trench whose near lip is 0.30 m in front of the base, plus the two things that
# must NOT read as one: a rail of the same size going up, and a ray that hits no mesh.
def flat(x):    return 0.0
def trench(x):  return -1.0 if 0.295 <= x <= 0.555 else 0.0
def rail(x):    return 0.30 if 0.295 <= x <= 0.555 else 0.0
def void(x):    return float("-inf")

# Forward window: x in {0.1 .. 0.7} is 7 columns, |y| <= 0.3 is 7 rows, so 49 rays, of
# which the trench covers x in {0.3, 0.4, 0.5} -> 3 columns -> 21.
AHEAD_FRAC = 21 / 49
# Under window: x in {-0.4 .. 0.4} is 9 columns; the trench covers {0.3, 0.4} -> 2.
UNDER_FRAC = 2 / 9

_failed = []


def check(label, got, want):
    ok = abs(float(got) - float(want)) < 1e-4
    if not ok:
        _failed.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label:<56} got {float(got):+.4f}  want {want:+.4f}")


def main() -> int:
    frac = lambda ground, win: se._gap_fraction(make_env(ground), SCAN, DEPTH, win, Y_HALF)
    check("gap ahead: trench in the forward window", frac(trench, AHEAD), AHEAD_FRAC)
    check("gap under: same trench, centred window", frac(trench, UNDER), UNDER_FRAC)
    check("flat ground is not a gap", frac(flat, AHEAD), 0.0)
    check("a rail goes UP and is not a gap", frac(rail, AHEAD), 0.0)
    check("a ray that hits nothing is not a gap", frac(void, AHEAD), 0.0)

    kw = dict(depth=DEPTH, x_range=AHEAD, y_half=Y_HALF)
    takeoff = lambda g, v: se.gap_takeoff(make_env(g, vel=v), SCAN, ROBOT, **kw)
    check("takeoff: rising at 2 m/s into a trench, running", takeoff(trench, (1.0, 0, 2.0)),
          AHEAD_FRAC * 2.0)
    check("takeoff: the same jump on flat ground pays nothing", takeoff(flat, (1.0, 0, 2.0)), 0.0)
    check("takeoff: pogo at the lip, no forward speed", takeoff(trench, (0.0, 0, 2.0)), 0.0)
    check("takeoff: half of min_speed ramps to half", takeoff(trench, (0.25, 0, 2.0)),
          AHEAD_FRAC * 2.0 * 0.5)
    check("takeoff: coming back DOWN pays nothing", takeoff(trench, (1.0, 0, -2.0)), 0.0)
    check("takeoff: vz is capped at 2.5", takeoff(trench, (1.0, 0, 9.0)), AHEAD_FRAC * 2.5)
    # The exploit that forced world-frame vz. In the base frame a 30-degree nose-up rear at
    # 2 m/s reads +1.0 m/s of "vertical" velocity with every foot still on the ground.
    check("takeoff: rearing up while running is not a jump", takeoff(trench, (2.0, 0, 0.0)), 0.0)

    kwf = dict(depth=DEPTH, x_range=UNDER, y_half=Y_HALF)
    flight = lambda g, v, a: se.gap_flight(make_env(g, vel=v, air=a), SCAN, CONTACT, ROBOT, **kwf)
    up, one_down = (0.2, 0.2, 0.2, 0.2), (0.2, 0.0, 0.2, 0.2)
    check("flight: airborne over a trench at 1.5 m/s", flight(trench, (1.5, 0, 0), up),
          UNDER_FRAC * 1.5)
    check("flight: one foot down is a stride, not a flight", flight(trench, (1.5, 0, 0), one_down), 0.0)
    check("flight: airborne over flat ground pays nothing", flight(flat, (1.5, 0, 0), up), 0.0)
    check("flight: airborne but going backwards pays nothing", flight(trench, (-1.5, 0, 0), up), 0.0)

    print(f"\n{'ALL PASS' if not _failed else str(len(_failed)) + ' FAILED: ' + ', '.join(_failed)}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
