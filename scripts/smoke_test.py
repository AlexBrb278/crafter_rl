"""Smoke test: step Crafter (numpy) and Craftax-Classic (JAX, GPU) and report throughput.

JAX has no CUDA wheels for native Windows, so there Craftax runs on CPU; use e.g.
`--craftax_envs 64 --craftax_steps 200` to keep the CPU check short.
"""
import argparse
import time

import numpy as np


def test_crafter(steps=2000):
    import crafter
    env = crafter.Env(seed=0)
    obs = env.reset()
    t = time.time()
    for _ in range(steps):
        obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
        if done:
            env.reset()
    sps = steps / (time.time() - t)
    print(f"Crafter: obs {obs.shape} {obs.dtype}, actions {env.action_space.n}, "
          f"{len(info['achievements'])} achievements, {sps:.0f} steps/s (1 env, CPU)")


def test_craftax(num_envs=1024, steps=1000):
    import jax
    from craftax.craftax_env import make_craftax_env_from_name
    print("JAX devices:", jax.devices())
    for name in ["Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"]:
        env = make_craftax_env_from_name(name, auto_reset=True)
        params = env.default_params
        rng = jax.random.PRNGKey(0)
        obs, state = jax.vmap(env.reset, (0, None))(jax.random.split(rng, num_envs), params)
        n_act = env.action_space(params).n

        @jax.jit
        def rollout(rng, state):
            def body(carry, _):
                rng, state = carry
                rng, a_rng, s_rng = jax.random.split(rng, 3)
                act = jax.random.randint(a_rng, (num_envs,), 0, n_act)
                obs, state, r, d, info = jax.vmap(env.step, (0, 0, 0, None))(
                    jax.random.split(s_rng, num_envs), state, act, params)
                return (rng, state), r
            return jax.lax.scan(body, (rng, state), None, steps)

        jax.block_until_ready(rollout(rng, state))  # compile
        t = time.time()
        jax.block_until_ready(rollout(rng, state))
        sps = num_envs * steps / (time.time() - t)
        print(f"{name}: obs {obs.shape[1:]}, actions {n_act}, {sps/1e6:.3f}M steps/s ({num_envs} envs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--craftax_envs", type=int, default=1024)
    ap.add_argument("--craftax_steps", type=int, default=1000)
    args = ap.parse_args()
    test_crafter()
    test_craftax(args.craftax_envs, args.craftax_steps)
