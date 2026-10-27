"""Chapter 12: advanced actor-critic methods. Continuous control with DDPG, TD3, SAC and PPO.

Every earlier chapter picked one of a few buttons. Here the action is a number (how
hard to push), so "argmax over actions" is gone and the actor outputs the action itself.

  * DDPG  - a deterministic actor a = mu(s) and a Q critic; the actor climbs the critic's
            slope, dQ/da. Off-policy, replay buffer, target networks (chapter 9's tricks).
  * TD3   - DDPG's three fixes: twin critics (use the smaller, against over-optimism),
            noise on the target action (smooth the critic), and a delayed actor.
  * SAC   - a stochastic actor (a squashed bell curve) that is also rewarded for staying
            random (entropy), with the reward-vs-randomness balance tuned automatically.
  * PPO   - on-policy: collect a batch, then improve the policy but never let it move too
            far from the one that collected the data (the clipped ratio).

Part 1 races all four on Pendulum (swing a pole up and hold it). Part 2 lets SAC teach a
MuJoCo HalfCheetah to run, filmed at checkpoints.

    python -m ch12_actor_critic.actor_critic                 # both parts
    python -m ch12_actor_critic.actor_critic --cheetah-steps 0   # Pendulum only
"""

import faulthandler
import os

import sys

faulthandler.enable()  # a native crash in a worker (MuJoCo, GL) prints its stack instead of vanishing
# In pods MuJoCo renders through Mesa's OSMesa, which loads Mesa's own LLVM. The first
# torch.optim.Adam() lazily imports torch._dynamo, which probes for triton, whose native
# library bundles a different LLVM: the clash segfaulted the HalfCheetah worker on load.
# We never use torch.compile, so make `import triton` fail; torch then treats it as absent.
sys.modules.setdefault("triton", None)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # see ch09_dqn/dqn.py: one thread per learner process
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import base64
import copy
import io
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import asciichart
import deepviz
import viz
from ch09_dqn.dqn import usable_cores

OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))
torch.set_num_threads(1)
LEARNERS = ["DDPG", "TD3", "SAC", "PPO"]


def mlp(n_in, n_out, hidden=256, out_act=None):
    layers = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out)]
    return nn.Sequential(*layers, *([out_act] if out_act else []))


class Critic(nn.Module):
    """Q(s, a): the action goes IN, because it is a number, not one of a few outputs."""

    def __init__(self, n_obs, n_act):
        super().__init__()
        self.q = mlp(n_obs + n_act, 1)

    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1)).squeeze(-1)


class Actor(nn.Module):
    """Deterministic actor for DDPG/TD3: a = high * tanh(mu(s))."""

    def __init__(self, n_obs, n_act, high):
        super().__init__()
        self.mu, self.high = mlp(n_obs, n_act, out_act=nn.Tanh()), high

    def forward(self, s):
        return self.high * self.mu(s)


class GaussianActor(nn.Module):
    """SAC's actor: a bell curve per action dimension, squashed through tanh into range."""

    def __init__(self, n_obs, n_act, high):
        super().__init__()
        self.body = mlp(n_obs, 2 * n_act)
        self.high, self.n_act = high, n_act

    def forward(self, s, deterministic=False):
        mean, log_std = self.body(s).chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        u = mean if deterministic else mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        # log-prob of the squashed sample: Gaussian log-prob minus tanh's change of volume
        logp = (torch.distributions.Normal(mean, std).log_prob(u) - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
        return self.high * a, logp, mean, std


class Replay:
    def __init__(self, size, n_obs, n_act):
        self.S, self.S2 = np.zeros((size, n_obs), np.float32), np.zeros((size, n_obs), np.float32)
        self.A, self.R, self.T = np.zeros((size, n_act), np.float32), np.zeros(size, np.float32), np.zeros(size, np.float32)
        self.size, self.i, self.n = size, 0, 0

    def add(self, s, a, r, s2, t):
        self.S[self.i], self.A[self.i], self.R[self.i], self.S2[self.i], self.T[self.i] = s, a, r, s2, t
        self.i, self.n = (self.i + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, k, rng):
        idx = rng.integers(self.n, size=k)
        return (torch.from_numpy(x[idx]) for x in (self.S, self.A, self.R, self.S2, self.T))


def soft_update(target, net, tau=0.005):
    with torch.no_grad():
        for t, p in zip(target.parameters(), net.parameters()):
            t.mul_(1 - tau).add_(tau * p)


def evaluate(policy, env_id, episodes=3, seed=70_000):
    env = gym.make(env_id)
    total = 0.0
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(policy(s))
            total += r
            done = term or trunc
    return total / episodes


def train_off_policy(algo, env_id, budget, seed=0, gamma=0.99, lr=3e-4, batch=256, warmup=1000,
                     eval_every=2000, snap_at=(), noise=0.1):
    """DDPG, TD3 and SAC share this loop; they differ in the target and the actor update."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make(env_id)
    n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
    high = float(env.action_space.high[0])
    actor = GaussianActor(n_obs, n_act, high) if algo == "SAC" else Actor(n_obs, n_act, high)
    q1, q2 = Critic(n_obs, n_act), Critic(n_obs, n_act)
    actor_t, q1_t, q2_t = copy.deepcopy(actor), copy.deepcopy(q1), copy.deepcopy(q2)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + (list(q2.parameters()) if algo != "DDPG" else []), lr=lr)
    log_alpha = torch.zeros(1, requires_grad=True)  # SAC's temperature, tuned to hit a target entropy
    opt_al = torch.optim.Adam([log_alpha], lr=lr)
    mem = Replay(1_000_000, n_obs, n_act)
    hist, snaps, alphas, s = [], {}, [], None
    s, _ = env.reset(seed=seed)

    def act(x, explore):
        st = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            if algo == "SAC":
                return actor(st, deterministic=not explore)[0].numpy()
            a = actor(st).numpy()
        if explore:
            a = a + rng.normal(0, noise * high, n_act)
        return np.clip(a, -high, high)

    for step in range(1, budget + 1):
        a = rng.uniform(-high, high, n_act) if step <= warmup else act(s, True)
        s2, r, term, trunc, _ = env.step(a)
        mem.add(s, a, r, s2, float(term))
        s = s2
        if term or trunc:
            s, _ = env.reset()
        if step > warmup:
            S, A, R, S2, T = mem.sample(batch, rng)
            with torch.no_grad():
                if algo == "SAC":
                    a2, logp2, _, _ = actor(S2)
                    alpha = log_alpha.exp()
                    nxt = torch.min(q1_t(S2, a2), q2_t(S2, a2)) - alpha * logp2
                elif algo == "TD3":
                    eps = (torch.randn_like(A) * 0.2 * high).clamp(-0.5 * high, 0.5 * high)
                    a2 = (actor_t(S2) + eps).clamp(-high, high)  # target policy smoothing
                    nxt = torch.min(q1_t(S2, a2), q2_t(S2, a2))  # twin critics: take the smaller
                else:
                    nxt = q1_t(S2, actor_t(S2))
                y = R + gamma * (1 - T) * nxt
            loss_q = ((q1(S, A) - y) ** 2).mean() + (((q2(S, A) - y) ** 2).mean() if algo != "DDPG" else 0)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            if algo == "SAC":
                a_new, logp, _, _ = actor(S)
                loss_a = (log_alpha.exp().detach() * logp - torch.min(q1(S, a_new), q2(S, a_new))).mean()
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()
                loss_al = -(log_alpha * (logp.detach() - n_act)).mean()  # target entropy = -n_act
                opt_al.zero_grad()
                loss_al.backward()
                opt_al.step()
                soft_update(q1_t, q1), soft_update(q2_t, q2)
                if step % 1000 == 0:
                    alphas.append((step, log_alpha.exp().item()))
            elif algo == "DDPG" or step % 2 == 0:  # TD3 delays the actor: every 2nd step
                loss_a = -q1(S, actor(S)).mean()  # climb the critic's slope
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()
                soft_update(actor_t, actor), soft_update(q1_t, q1)
                if algo == "TD3":
                    soft_update(q2_t, q2)
        if step % eval_every == 0 or step == budget:
            hist.append((step, evaluate(lambda x: act(x, False), env_id)))
        if step in snap_at:
            snaps[step] = copy.deepcopy(actor.state_dict())
    snaps[budget] = copy.deepcopy(actor.state_dict())
    return hist, snaps, {"alpha": alphas, "q": (q1.state_dict(), q2.state_dict())}


class PPOPolicy(nn.Module):
    def __init__(self, n_obs, n_act, high):
        super().__init__()
        self.mu = mlp(n_obs, n_act, hidden=64)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))
        self.v = mlp(n_obs, 1, hidden=64)
        self.high = high

    def dist(self, s):
        return torch.distributions.Normal(self.mu(s), self.log_std.exp())


def train_ppo(env_id, budget, seed=0, gamma=0.99, lam=0.95, lr=3e-4, n_envs=8, n_steps=256, epochs=10,
              minibatch=256, clip_eps=0.2, eval_every=20_000, snap_at=()):
    """PPO: collect a batch with the current policy, then 10 epochs of improvement that
    are not allowed to move any action's probability by more than about 20%."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = gym.make_vec(env_id, num_envs=n_envs, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    n_obs, n_act = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    high = float(envs.single_action_space.high[0])
    pol = PPOPolicy(n_obs, n_act, high)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    s, _ = envs.reset(seed=[seed * 1000 + i for i in range(n_envs)])
    hist, snaps, steps, next_eval, clip_frac = [], {}, 0, eval_every, []
    snap_at = sorted(snap_at)
    while steps < budget:
        S, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(n_steps):
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                d = pol.dist(st)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = pol.v(st).squeeze(-1)
            s2, r, term, trunc, info = envs.step(np.clip(a.numpy(), -high, high))
            boot = np.zeros(n_envs, np.float32)
            if trunc.any():  # a time limit is not a failure: bootstrap from the real final state
                fo = np.stack([info["final_obs"][i] if trunc[i] else s2[i] for i in range(n_envs)])
                with torch.no_grad():
                    boot = np.where(trunc & ~term, pol.v(torch.as_tensor(fo, dtype=torch.float32)).squeeze(-1).numpy(), 0)
            S.append(s), A.append(a.numpy()), LP.append(lp.numpy()), R.append(r + gamma * boot)
            D.append(term | trunc), V.append(v.numpy())
            s = s2
            steps += n_envs
        with torch.no_grad():
            last_v = pol.v(torch.as_tensor(s, dtype=torch.float32)).squeeze(-1).numpy()
        R, D, V = np.array(R, np.float32), np.array(D, np.float32), np.array(V)
        adv, g = np.zeros_like(R), np.zeros(n_envs, np.float32)
        for t in reversed(range(n_steps)):
            nv = last_v if t == n_steps - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + V
        flat = lambda x, *sh: torch.as_tensor(np.array(x).reshape(-1, *sh), dtype=torch.float32)
        S_t, A_t, LP_t = flat(S, n_obs), flat(A, n_act), flat(LP)
        ADV, RET = flat(adv), flat(ret)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(epochs):
            for idx in torch.randperm(len(S_t)).split(minibatch):
                d = pol.dist(S_t[idx])
                ratio = (d.log_prob(A_t[idx]).sum(-1) - LP_t[idx]).exp()
                # The clipped objective: a gain is only counted while the new policy is within
                # 20% of the old one on this action; past that, there is nothing to gain by moving.
                obj = torch.min(ratio * ADV[idx], ratio.clamp(1 - clip_eps, 1 + clip_eps) * ADV[idx])
                loss = -obj.mean() + 0.5 * ((pol.v(S_t[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
                clip_frac.append(float(((ratio - 1).abs() > clip_eps).float().mean()))
        while steps >= next_eval:
            hist.append((next_eval, evaluate(lambda x: np.clip(pol.mu(torch.as_tensor(x, dtype=torch.float32)).detach().numpy(), -high, high), env_id)))
            next_eval += eval_every
        while snap_at and steps >= snap_at[0]:
            snaps[snap_at.pop(0)] = copy.deepcopy(pol.state_dict())
    if not hist or hist[-1][0] < steps:  # always score the final policy
        hist.append((steps, evaluate(lambda x: np.clip(pol.mu(torch.as_tensor(x, dtype=torch.float32)).detach().numpy(), -high, high), env_id)))
    snaps[budget] = copy.deepcopy(pol.state_dict())
    return hist, snaps, {"clip_frac": float(np.mean(clip_frac[-200:])) if clip_frac else 0.0}


def train(learner, env_id, budget, seed, snap_at=()):
    if learner == "PPO":
        return train_ppo(env_id, budget, seed, snap_at=snap_at)
    return train_off_policy(learner, env_id, budget, seed, snap_at=snap_at)


# ── main ────────────────────────────────────────────────────────────────────────

def _late(hist):
    v = [h[1] for h in hist]
    return float(np.mean(v[len(v) * 2 // 3:]))


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-steps", type=int, default=30_000, help="Pendulum steps for DDPG/TD3/SAC")
    ap.add_argument("--ppo-steps", type=int, default=400_000, help="Pendulum steps for PPO")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cheetah-steps", type=int, default=150_000, help="SAC on HalfCheetah (0 = skip)")
    args = ap.parse_args(argv)

    budget = {l: (args.ppo_steps if l == "PPO" else args.off_steps) for l in LEARNERS}
    snap = {l: (budget[l] // 10, budget[l] // 3) for l in LEARNERS}
    cheetah_snaps = tuple(int(args.cheetah_steps * f) for f in (0.05, 0.25, 0.5)) if args.cheetah_steps else ()
    t = time.perf_counter()
    with ProcessPoolExecutor(usable_cores(), mp_context=mp.get_context("spawn")) as ex:
        cheetah_f = (ex.submit(train_off_policy, "SAC", "HalfCheetah-v5", args.cheetah_steps, 0,
                               eval_every=5000, snap_at=cheetah_snaps) if args.cheetah_steps else None)
        futures = {(l, s): ex.submit(train, l, "Pendulum-v1", budget[l], s, snap[l] if s == 0 else ())
                   for s in range(args.seeds) for l in LEARNERS}
        runs = {k: f.result() for k, f in futures.items()}
        cheetah = cheetah_f.result() if cheetah_f else None
    print(f"trained {len(futures)} Pendulum runs ({args.seeds} seeds each)"
          + (f" + SAC on HalfCheetah ({args.cheetah_steps:,} steps)" if cheetah else "")
          + f" in {time.perf_counter() - t:.0f}s\n")

    stats = {}
    for l in LEARNERS:
        late = np.array([_late(runs[l, s][0]) for s in range(args.seeds)])
        curves = np.array([[h[1] for h in runs[l, s][0]] for s in range(args.seeds)])
        reach = []
        for s in range(args.seeds):
            hit = [h[0] for h in runs[l, s][0] if h[1] >= -250]
            reach.append(hit[0] if hit else None)
        ok = [r for r in reach if r]
        stats[l] = {"late": late, "mean": float(late.mean()), "sem": float(late.std(ddof=1) / np.sqrt(len(late))),
                    "curves": curves, "x": [h[0] for h in runs[l, 0][0]],
                    "reach": float(np.median(ok)) if ok else None, "reached": len(ok)}
        rch = f"{stats[l]['reach']:,.0f}" if ok else "never"
        print(f"{l:>5}: Pendulum score, last third {stats[l]['mean']:7.1f} ± {stats[l]['sem']:5.1f}   "
              f"first holds it (-250 or better) at step {rch} (median, {len(ok)}/{args.seeds} seeds)")
    if cheetah:
        print(f"\nSAC HalfCheetah: final evaluation {cheetah[0][-1][1]:.0f} "
              f"(random ~ -300; a good runner 3,000+)")

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2 if cheetah else 1, figsize=(15 if cheetah else 9, 4.6), squeeze=False)
    for l, st in stats.items():
        m = st["curves"].mean(axis=0)
        line, = ax[0, 0].plot(st["x"], m, label=l)
        lo, hi = np.percentile(st["curves"], [25, 75], axis=0)
        ax[0, 0].fill_between(st["x"], lo, hi, color=line.get_color(), alpha=0.12)
    ax[0, 0].set(xscale="log", xlabel="environment steps (log scale)", ylabel="evaluation score",
                 title=f"Pendulum: mean of {args.seeds} seeds (band: middle half)", ylim=(-1600, 0))
    ax[0, 0].axhline(-250, color="k", ls="--", lw=1, label="holding it up")
    ax[0, 0].legend(fontsize=9)
    if cheetah:
        ax[0, 1].plot([h[0] for h in cheetah[0]], [h[1] for h in cheetah[0]], color="#e17055")
        ax[0, 1].set(xlabel="environment steps", ylabel="evaluation score", title="SAC on MuJoCo HalfCheetah")
    fig.suptitle("Chapter 12: continuous control")
    fig.tight_layout()
    path = OUT / "actor_critic.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(asciichart.bars({l: st["mean"] for l, st in stats.items()}, title="Pendulum, last-third score (higher is better)", fmt="{:.0f}"))
    print(f"\nwrote {path}")
    return _report(runs, stats, cheetah, path, args, budget)


# ── the report ──────────────────────────────────────────────────────────────────

def _actor(l, state):
    n_obs, n_act, high = 3, 1, 2.0
    net = (GaussianActor if l == "SAC" else PPOPolicy if l == "PPO" else Actor)(n_obs, n_act, high)
    net.load_state_dict(state)
    return net


def record_pendulum(l, net, seed=11, critics=None):
    """One deterministic swing-up, with the actor's torque (and SAC's spread, TD3's two critics)."""
    env = gym.make("Pendulum-v1")
    s, _ = env.reset(seed=seed)
    frames, total, done = [], 0.0, False
    q1 = q2 = None
    if critics:
        q1, q2 = Critic(3, 1), Critic(3, 1)
        q1.load_state_dict(critics[0]), q2.load_state_dict(critics[1])
    while not done:
        st = torch.as_tensor(s, dtype=torch.float32)
        f = {"th": round(float(np.arctan2(s[1], s[0])), 3)}
        with torch.no_grad():
            if l == "SAC":
                a, _, mean, std = net(st, deterministic=True)
                f["mean"], f["std"] = round(float(2 * torch.tanh(mean)), 3), round(float(std), 3)
            elif l == "PPO":
                a = net.mu(st).clamp(-2, 2)
            else:
                a = net(st)
            if q1 is not None:
                at = a.reshape(1)
                f["q"] = [round(float(q1(st, at)), 1), round(float(q2(st, at)), 1)]
        f["u"] = round(float(a.reshape(-1)[0]), 3)
        f["score"] = round(total, 1)
        frames.append(f)
        s, r, term, trunc, _ = env.step(a.reshape(1).numpy())
        total += r
        done = term or trunc
    return frames, total


def _video(frames, fps=30):
    import av
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    st = out.add_stream("h264", rate=fps)
    st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    for fr in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    return base64.b64encode(buf.getvalue()).decode()


def film_cheetah(state, seconds=6, seed=3):
    # The model's "track" camera follows the body; the default camera is fixed, and a
    # trained cheetah runs straight out of its frame within a second.
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array", width=320, height=200, camera_name="track")
    net = GaussianActor(17, 6, 1.0)
    net.load_state_dict(state)
    s, _ = env.reset(seed=seed)
    x0 = float(env.unwrapped.data.qpos[0])
    frames, total = [], 0.0
    for _ in range(int(seconds * 20)):  # HalfCheetah runs at 20 control steps per second
        with torch.no_grad():
            a = net(torch.as_tensor(s, dtype=torch.float32), deterministic=True)[0].numpy()
        s, r, term, trunc, _ = env.step(a)
        total += r
        frames.append(env.render())
    dist = float(env.unwrapped.data.qpos[0]) - x0
    env.close()
    return _video(frames, fps=20), total, dist


def _report(runs, stats, cheetah, path, args, budget):
    flights = {}
    for l in LEARNERS:
        _, snaps, extra = runs[l, 0]
        net = _actor(l, snaps[budget[l]])
        flights[l] = record_pendulum(l, net, critics=extra["q"] if l == "TD3" else None)
    # Measured from SAC's recorded swing-up, not assumed.
    sac = flights["SAC"][0]
    up = [np.cos(f["th"]) > 0.95 for f in sac]
    swing = [f for f, u in zip(sac, up) if not u]
    held = [f for f, u in zip(sac, up) if u]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    full = mean([abs(f["u"]) > 1.9 for f in swing])
    sd_swing, sd_held = mean([f["std"] for f in swing]), mean([f["std"] for f in held])
    u_swing, u_held = mean([abs(f["u"]) for f in swing]), mean([abs(f["u"]) for f in held])
    spread_txt = ("narrower while swinging" if sd_swing < sd_held else "wider while swinging")

    def row(l, st):
        reach = f"{st['reach']:,.0f}" if st["reach"] else "never"
        return (f"<tr><td>{l}</td><td>{budget[l]:,}</td><td>{st['mean']:.0f} ± {st['sem']:.0f}</td>"
                f"<td>{reach} ({st['reached']}/{args.seeds} seeds)</td></tr>")

    rows = "".join(row(l, st) for l, st in stats.items())
    alpha = runs["SAC", 0][2]["alpha"]
    parts = [
        "<p style='font-size:16px'><b>In one sentence:</b> when the action is a number (how hard "
        "to push) instead of a button, the actor has to output the action itself, and the "
        "critic tells it which way to nudge it.</p>",
        '<div class="card"><b>Pendulum.</b> A pole hangs from a motor at its top end. The agent '
        "chooses a torque between −2 and +2 every step (any value, not one of a few buttons). It "
        "starts at a random angle, usually hanging down, and is penalised every step for being "
        "away from upright, for spinning, and for using torque. The motor is too weak to lift the "
        "pole straight up: it has to <b>pump</b>, swinging back and forth to build speed, then "
        "catch it at the top and hold it. A score around −150 means it swings up fast and holds; "
        "a random agent scores about −1,200.</div>",

        "<h3>1. Four continuous-control learners</h3>",
        viz.figure(path),
        f"<table class='dc'><tr><th>learner</th><th>steps it trained for</th><th>last-third score</th>"
        f"<th>first holds the pole (median step)</th></tr>{rows}</table>",
        "<pre>DDPG   actor a = μ(s);  critic Q(s, a);  actor climbs ∂Q/∂a           (chapter 9's tricks, continuous)\n"
        "TD3    DDPG + twin critics (use the smaller) + noisy target action + actor updates every 2nd step\n"
        "SAC    actor = a bell curve over torques; reward + α·(how random it is); α tuned automatically\n"
        "PPO    on-policy: collect 2,048 moves, improve for 10 passes, but clip any move of more than 20%</pre>",
        viz.explain(
            how=f"Note the log-scale x-axis. DDPG, TD3 and SAC are <b>off-policy</b>: every move goes "
                f"into a replay buffer and is learned from many times, so they got {args.off_steps:,} "
                f"steps. PPO is <b>on-policy</b>: each batch of moves is used for one round of "
                f"updates and thrown away, so it needs far more; it got {args.ppo_steps:,}.",
            notice="Off-policy learners are dramatically more sample-efficient: compare the step "
                   "counts in the last column. PPO's strengths are elsewhere: it is stable, simple to "
                   "tune, and parallelizes across many simulators, which is why it trains robots in "
                   "fast simulators (rl-mujoco in this repo uses it for a humanoid)."),

        "<h3>2. Inside each actor: the torque is a dial now</h3>",
        viz.row(*[deepviz.pendulum_player(fr, f"{l}: score {tot:.0f}") for l, (fr, tot) in flights.items()]),
        viz.explain(
            how="The orange arc is the torque the actor chose this frame (left = anticlockwise, the "
                "full arc = the motor's limit). For SAC, the shaded band is its bell curve's spread: "
                "how unsure it is. For TD3, the panel shows both critics' opinions of the chosen "
                "action; it trusts the lower one.",
            notice=f"Watch the pump. Measured on SAC's flight: while swinging it uses full torque "
                   f"{full:.0%} of the time (average |torque| {u_swing:.2f}); once the pole is up, "
                   f"only small corrections (average {u_held:.2f}). SAC's bell curve is {spread_txt} "
                   f"(spread {sd_swing:.2f} swinging vs {sd_held:.2f} balanced): mid-swing the right "
                   "answer is obvious, full power in the direction of motion, while balanced at the "
                   "top many small torques are about equally good, so it stays less sure."),
        "<h3>3. SAC's thermostat for randomness</h3>",
        _alpha_picture(alpha),
        viz.explain(
            how="SAC adds α × (how random the policy is) to the reward, and adjusts α itself so the "
                "policy keeps a target amount of randomness.",
            notice="α starts high (explore!) and falls as the policy finds out what works: the "
                   "agent automatically spends less effort on exploring once it knows more."),
    ]
    if cheetah:
        hist, snaps, _ = cheetah
        vids = []
        for step in sorted(snaps):
            b64, total, dist = film_cheetah(snaps[step])
            vids.append(f'<div style="display:inline-block;margin:4px;text-align:center"><video src="data:video/mp4;base64,{b64}" '
                        f'controls loop muted autoplay playsinline style="width:300px"></video>'
                        f'<div style="font-size:12px">after {step:,} steps: ran {dist:+.1f} m in 6 s '
                        f'({total:.0f} points)</div></div>')
        parts += [
            "<h3>4. The same idea on a real robot body: SAC teaches HalfCheetah to run</h3>",
            '<div class="card"><b>HalfCheetah</b> (MuJoCo physics) is a two-legged, cat-like body '
            "with six motorized joints (back and front thigh, shin, foot). Every step the actor sets "
            "all six torques at once; the reward is forward speed minus a little for effort. Nothing "
            "tells it what running looks like.</div>",
            "".join(vids),
            viz.explain(how="The camera follows the body, so watch the floor tiles slide past for "
                            "speed, and read the distance under each clip.",
                        notice=f"From flailing to a gait in {args.cheetah_steps:,} steps (about "
                               f"{args.cheetah_steps / 20 / 60:.0f} simulated minutes), with the same SAC "
                               "code as the pendulum, just six dials instead of one. Final evaluation: "
                               f"{hist[-1][1]:.0f} points per 1,000-step episode."),
        ]
    parts.append(viz.takeaways([
        "Continuous actions: the actor outputs the action itself; the critic says which way to nudge it.",
        "DDPG brings DQN's replay and target networks to continuous control; TD3 fixes its over-optimism.",
        "SAC keeps the policy deliberately random, tuning how much automatically.",
        "PPO is on-policy: hungrier for samples, but stable and easy to scale across simulators.",
        "The same algorithms that swing a pendulum teach a simulated body to run.",
        "Bonus: <code>flyte run pipeline.py ch12_bonus_humanoid</code> scales this PPO to 4,096 GPU "
        "simulators and teaches a humanoid robot to walk.",
    ]))
    return parts


def _alpha_picture(alpha):
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    ax.plot([a[0] for a in alpha], [a[1] for a in alpha], color="#fdcb6e")
    ax.set_yscale("log")
    ax.set_xlabel("training step", color="#ccc")
    ax.set_ylabel("α (temperature)", color="#ccc")
    ax.tick_params(colors="#888")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


if __name__ == "__main__":
    main()
