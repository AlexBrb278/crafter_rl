"""Run upstream Craftax_Baselines PPO-RNN unchanged, plus local logging.

The upstream code only logs to Weights & Biases. We run wandb offline and also
append every logged update to <run_dir>/metrics.jsonl so results can be read
without a wandb account.

Usage (same flags as upstream ppo_rnn.py, plus --run_dir):
    python methods/01_ppo_rnn/train.py --run_dir runs/ppo_rnn/classic_s0 \
        --env_name Craftax-Classic-Symbolic-v1 --total_timesteps 1e9 --seed 0
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")


def main():
    argv = sys.argv[1:]
    if "--run_dir" not in argv:
        sys.exit("--run_dir is required")
    i = argv.index("--run_dir")
    run_dir = os.path.abspath(argv[i + 1])
    del argv[i : i + 2]
    os.makedirs(run_dir, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["WANDB_DIR"] = run_dir
    sys.path.insert(0, UPSTREAM)

    import numpy as np
    from wandb.sdk.wandb_run import Run

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    original_log = Run.log
    t0 = time.time()
    step = {"n": 0}

    # wandb.init() rebinds the module-level wandb.log to run.log, so patch the class.
    def log_locally(self, data, *args, **kwargs):
        step["n"] += 1
        row = {"update": step["n"], "wall_s": round(time.time() - t0, 2)}
        for k, v in data.items():
            v = np.asarray(v)
            row[k] = float(v.mean()) if v.size else None
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return original_log(self, data, *args, **kwargs)

    Run.log = log_locally

    with open(os.path.join(run_dir, "cmd.txt"), "w") as f:
        f.write(" ".join([sys.executable, "ppo_rnn.py"] + argv) + "\n")

    import runpy

    sys.argv = [os.path.join(UPSTREAM, "ppo_rnn.py")] + argv
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
