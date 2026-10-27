"""The G1 environment, and the reward knobs we actually turn on stream.

We do NOT hand-roll the humanoid env here. `mujoco_playground` ships DeepMind's own
`G1JoystickFlatTerrain`, which is the config their published G1 walking results come
from, and starting from a known-good baseline is the entire point: the workshop
tutorial's hand-written `G1RewardWrapper` got to "stands and shuffles" and stopped,
and it was never clear whether the reward or the sample count was at fault. Start
from something that demonstrably walks, THEN turn knobs.

── What the env actually looks like ────────────────────────────────────────────
Measured on this box (mujoco 3.11.0 / playground registry, 2026-08-04):

    action_size       29                      (29 actuated joints)
    observation_size  {'state': (103,),       <- what the POLICY sees
                       'privileged_state': (216,)}   <- what the VALUE net sees

That dict is not incidental, it is asymmetric actor-critic: the critic is allowed
ground-truth simulator state (contact forces, true body velocities) that a real robot
could never measure, while the policy only sees onboard-sensor-shaped observations.
It makes the value estimate much easier to learn without making the policy
un-deployable. It is also why `train.py` must pass `policy_obs_key="state"` and
`value_obs_key="privileged_state"` into the network factory. Get those backwards and
you either cripple the critic or train a policy that cannot exist on hardware.

── The joystick part ───────────────────────────────────────────────────────────
"Joystick" means the env samples a random velocity command each episode (x in
[-1, 1] m/s, y in [-0.5, 0.5], yaw in [-1, 1] rad/s) and rewards TRACKING it. So the
policy is not learning "walk forward", it is learning "go the speed and direction I
am told", which is both harder and much more useful: the finished policy is
steerable, which makes for a far better demo than a robot that only walks north.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Terrain ─────────────────────────────────────────────────────────────────────
#
# Two Playground envs, same code path, same tuned PPO config, and the same
# observation/action shapes. They differ in the scene XML:
#
#   flat   scene_mjx_feetonly_flat_terrain.xml    a plane
#   rough  scene_mjx_feetonly_rough_terrain.xml   a 10x10m heightfield (hfield.png,
#                                                 0.05m relief) with a rocky texture
#
# Rough is the better demo on both counts. It is a genuinely harder problem, so the
# gait has to cope with ground that is not level, and it does not look like the robot
# is walking on nothing. Verified loading and rendering on this box 2026-08-04.
#
# Known cosmetic wart: the rough scene ships no skybox, so the background renders
# black where flat's is grey. Only affects the replay's looks, not the physics.
TERRAINS: dict[str, str] = {
    "flat": "G1JoystickFlatTerrain",
    "rough": "G1JoystickRoughTerrain",
}
DEFAULT_TERRAIN = "rough"

# Kept for callers that just want "the env": the flat one, which is what the first
# runs on this box trained against.
ENV_NAME = TERRAINS["flat"]


def env_name(terrain: str = DEFAULT_TERRAIN) -> str:
    try:
        return TERRAINS[terrain]
    except KeyError:
        raise ValueError(
            f"Unknown terrain {terrain!r}. Known: {', '.join(TERRAINS)}"
        ) from None

# Measured, not guessed. Asserted at env-build time so a Playground upgrade that
# changes the observation layout fails loudly here instead of silently training a
# mis-shaped network.
EXPECTED_ACTION_SIZE = 29
EXPECTED_OBS_SIZES = {"state": 103, "privileged_state": 216}

# The policy reads "state", the critic reads "privileged_state". See the docstring.
POLICY_OBS_KEY = "state"
VALUE_OBS_KEY = "privileged_state"


# ── The njmax fix ───────────────────────────────────────────────────────────────
#
# Playground ships `njmax=90` for this env. On this box that overflows almost
# immediately: the very first rollout printed
#
#     nefc overflow - please increase njmax to 93
#
# repeatedly. njmax caps the constraint-Jacobian rows the solver can hold, and a
# humanoid in contact with the ground generates more than 90 of them. On overflow MJX
# silently DROPS constraints rather than erroring, which means feet sink through the
# floor and the physics quietly stops being the physics you think you are training on.
#
# 128 gives headroom over the observed 93 without meaningfully costing memory (the
# constraint buffers are num_envs x njmax, so this is a few MB at 8192 envs).
NJMAX = 128


@dataclass(frozen=True)
class RewardPreset:
    """A named set of overrides applied on top of Playground's reward scales.

    Only the keys you name are changed; everything else keeps DeepMind's tuned value.
    That is deliberate. The failure mode in the workshop tutorial was writing a whole
    reward from scratch and then not knowing which of ten terms was wrong. Here the
    baseline is fixed and each preset is a small, legible diff you can point at.
    """
    key: str
    scales: dict[str, float] = field(default_factory=dict)
    # Non-reward config overrides (push_config, noise_config, command ranges, ...).
    overrides: dict[str, object] = field(default_factory=dict)
    notes: str = ""


PRESETS: dict[str, RewardPreset] = {
    # The control. DeepMind's config, untouched. Whatever this scores is the number
    # every other preset has to beat.
    "baseline": RewardPreset(
        key="baseline",
        notes="Playground's tuned G1 config, unmodified. The control.",
    ),

    # The tutorial's diagnosis was "it shuffles instead of stepping". Playground
    # already has the two terms that fix that, and one of them is OFF by default:
    #   feet_air_time   2.0  (on)  rewards a foot spending time in the air
    #   feet_clearance  0.0  (OFF) penalizes a MOVING foot for being away from
    #                              max_foot_height (0.15 m): sliding along the floor
    #                              costs, swinging at step height does not
    # Turning clearance on is the single most direct answer to shuffling.
    # It is a COST, so the scale must be negative (-0.5 is Playground's H1 humanoid
    # value; Go1 and Spot use -2.0). An earlier +1.0 here REWARDED feet far from step
    # height, sliding feet included: the opposite of the preset's purpose.
    "high-step": RewardPreset(
        key="high-step",
        scales={"feet_clearance": -0.5, "feet_air_time": 3.0},
        notes="Anti-shuffle: turn on foot clearance, push air time harder.",
    ),

    # The opposite bet: the tutorial also saw jittery, twitchy actions. Penalising
    # action rate and energy buys smoothness, at some cost in raw tracking.
    "smooth": RewardPreset(
        key="smooth",
        scales={"action_rate": -0.75, "energy": -0.001, "dof_acc": -2.5e-7},
        notes="Penalise jerk and energy. Slower, cleaner gait.",
    ),

    # Measured weakness of the 300M baseline run (2026-08-04): forward and backward
    # tracking are good (commanded +1.0 -> +0.88 m/s body-frame, commanded -0.5 ->
    # -0.43), but TURNING is essentially absent (commanded +1.0 rad/s yaw -> -0.06)
    # and strafing only half lands (+0.4 -> +0.22).
    #
    # The likely reason is that it is cheap to ignore: tracking_ang_vel is scaled 0.75
    # against tracking_lin_vel's 1.0 and a large pile of other terms, so a policy that
    # nails forward speed and ignores yaw still scores well. This preset roughly
    # doubles the turning term and lifts lateral tracking with it.
    "turner": RewardPreset(
        key="turner",
        scales={"tracking_ang_vel": 1.5, "tracking_lin_vel": 1.25},
        notes="Push yaw tracking, which the baseline barely learns.",
    ),

    # Easy mode, for the "does the harness work end to end" run and for the first
    # half of the stream. No random pushes and no sensor noise means the policy has a
    # much simpler problem, so it shows visible progress far sooner. It will NOT be
    # robust; that is the point of showing it next to baseline.
    "no-perturb": RewardPreset(
        key="no-perturb",
        overrides={"push_config.enable": False, "noise_config.level": 0.0},
        notes="Pushes and sensor noise off. Learns fastest, generalises worst.",
    ),
}

DEFAULT_PRESET = "baseline"


def get_preset(key: str) -> RewardPreset:
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(
            f"Unknown reward preset {key!r}. Known: {', '.join(PRESETS)}"
        ) from None


def _set_nested(cfg, dotted: str, value) -> None:
    """Set `a.b.c` on a ConfigDict. Raises if the path does not exist."""
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    if not hasattr(node, parts[-1]):
        raise KeyError(f"{dotted!r} is not a key in the env config")
    setattr(node, parts[-1], value)


def build_env_config(
    preset_key: str = DEFAULT_PRESET,
    episode_length: int | None = None,
    terrain: str = DEFAULT_TERRAIN,
):
    """Playground's default G1 config, plus the njmax fix and one preset's diff."""
    from mujoco_playground import registry

    preset = get_preset(preset_key)
    cfg = registry.get_default_config(env_name(terrain))

    cfg.njmax = NJMAX

    for name, value in preset.scales.items():
        if name not in cfg.reward_config.scales:
            raise KeyError(
                f"preset {preset.key!r} sets reward scale {name!r}, which this "
                f"version of Playground does not have. Known: "
                f"{', '.join(sorted(cfg.reward_config.scales))}"
            )
        cfg.reward_config.scales[name] = value

    for dotted, value in preset.overrides.items():
        _set_nested(cfg, dotted, value)

    if episode_length is not None:
        cfg.episode_length = episode_length

    return cfg


def build_env(
    preset_key: str = DEFAULT_PRESET,
    episode_length: int | None = None,
    terrain: str = DEFAULT_TERRAIN,
):
    """Load the G1 env with a preset applied, and verify its shapes are what we expect."""
    from mujoco_playground import registry

    name = env_name(terrain)
    cfg = build_env_config(preset_key, episode_length, terrain)
    env = registry.load(name, config=cfg)

    if env.action_size != EXPECTED_ACTION_SIZE:
        raise RuntimeError(
            f"{name} action_size is {env.action_size}, expected "
            f"{EXPECTED_ACTION_SIZE}. Playground changed the model; re-check the "
            f"network config in train.py before trusting a run."
        )
    obs = env.observation_size
    if not isinstance(obs, dict):
        raise RuntimeError(
            f"Expected a dict observation (asymmetric actor-critic), got {type(obs)}. "
            f"policy_obs_key/value_obs_key in train.py assume the dict form."
        )
    for key, want in EXPECTED_OBS_SIZES.items():
        got = obs.get(key)
        if got is None:
            raise RuntimeError(f"observation_size is missing {key!r}: got {obs}")
        if tuple(got)[0] != want:
            raise RuntimeError(
                f"observation_size[{key!r}] is {tuple(got)}, expected ({want},)."
            )
    return env, cfg


def get_randomizer(terrain: str = DEFAULT_TERRAIN):
    """Playground's domain randomizer for the G1, or None if it has none.

    Domain randomization (per-env friction, mass, motor gains) is what makes the
    policy survive contact with a physics it was not trained on. It costs nothing at
    training time here because the randomization is vmapped along with everything else.
    """
    from mujoco_playground import registry

    return registry.get_domain_randomizer(env_name(terrain))
