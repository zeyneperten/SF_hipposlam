"""Train/evaluate the real high-level RNN on completed two-arm choices.

Run: python -m sf_working_directories.zeynep.bandit.train --help
Only PyTorch, NumPy, and the Python standard library are required.
"""

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from sf_working_directories.zeynep.bandit.checkpoints import (
    build_model_from_checkpoint,
    load_checkpoint_into,
    save_checkpoint,
)
from sf_working_directories.zeynep.bandit.environment import BanditConfig, BlockSwitchBandit
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import (
    HighLevelContextRNN_Learner,
    TrialHistory,
    high_level_policy_loss,
    q_loss,
)


def initial_hidden(model, num_envs):
    """Match the full core's initial trigger, with no completed trial to credit."""
    parameter = next(model.parameters())
    h = parameter.new_zeros(num_envs, model.d_H)
    z = parameter.new_zeros(num_envs, model.K)
    event = torch.ones(num_envs, dtype=torch.bool, device=parameter.device)
    with torch.no_grad():
        return model.update_state(event, parameter.new_zeros(num_envs), h, z)


def trial_loss(model, history, samples):
    """Reconstruct decisions using only history preceding their own rewards.

    Each tuple contains (pre-outcome history, decision hidden, sampled mode,
    obtained reward). Replaying never samples a replacement action.
    """
    histories, hidden, modes, rewards = [torch.cat(values) for values in zip(*samples)]
    scores = history.scores_from_history(model, histories, hidden).unsqueeze(0)
    modes, rewards = modes.unsqueeze(0), rewards.unsqueeze(0)
    mask = torch.ones_like(rewards, dtype=torch.bool)
    if model.is_policy:
        return high_level_policy_loss(scores, modes, rewards, mask)[0]
    return q_loss(scores, modes, rewards, mask)


def policy_probabilities(model, scores):
    return torch.softmax(scores if model.is_policy else scores / model.tau, dim=-1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_revision():
    try:
        root = Path(__file__).resolve().parents[3]
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


@torch.no_grad()
def evaluate(model, task, num_sessions, seed, policy="sampled", reset_memory=False):
    """Frozen independent sessions; hidden task metadata only enters metrics.

    All policies get the same environmental random stream. Sampled and greedy
    are separate closed-loop runs, since each induces its own reward history.
    """
    env = BlockSwitchBandit(task, num_sessions, seed)
    device = next(model.parameters()).device
    event = torch.ones(num_sessions, dtype=torch.bool, device=device)
    h = initial_hidden(model, num_sessions)
    rng = np.random.default_rng(seed + 1)
    arm_to_mode = np.argsort(task.mode_to_arm)
    previous_arm = rng.integers(0, 2, num_sessions)
    previous_reward = np.ones(num_sessions)
    previous_high = np.full(num_sessions, -1)
    switch_age = np.full(num_sessions, -1)
    rows = []
    was_training = model.training
    model.eval()
    # Evaluation must not change the training action RNG stream.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + 2)
            for trial in range(env.session_trials):
                if reset_memory:
                    h = initial_hidden(model, num_sessions)
                scores = model.head(h)
                probabilities = policy_probabilities(model, scores)
                if policy == "sampled":
                    choices = torch.distributions.Categorical(probs=probabilities).sample().cpu().numpy()
                elif policy == "greedy":
                    choices = scores.argmax(-1).cpu().numpy()
                elif policy == "random":
                    choices = rng.integers(0, 2, num_sessions)
                elif policy == "win_stay_lose_shift":
                    choices = arm_to_mode[np.where(previous_reward > 0, previous_arm, 1 - previous_arm)]
                elif policy == "oracle":
                    choices = arm_to_mode[env.high_arm]
                else:
                    raise ValueError(f"Unknown evaluation policy: {policy}")
                reward, info = env.step(choices)
                switch_age = np.where(info["actual_switch"], 0, np.where(switch_age >= 0, switch_age + 1, -1))
                probs = probabilities.cpu().numpy()
                values = scores.cpu().numpy()
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).cpu().numpy()
                for index in range(num_sessions):
                    high = int(info["high_arm"][index])
                    chosen = int(info["chosen_arm"][index])
                    rows.append(
                        {
                            "session": index,
                            "trial": trial,
                            "block_index": int(info["block_index"][index]),
                            "trial_in_block": int(info["trial_in_block"][index]),
                            "nominal_boundary": bool(info["nominal_boundary"][index]),
                            "actual_switch": bool(info["actual_switch"][index]),
                            "since_actual_switch": int(switch_age[index]),
                            "mode": int(choices[index]),
                            "chosen_arm": chosen,
                            "high_arm": high,
                            "reward": float(reward[index]),
                            "high_arm_choice": int(chosen == high),
                            "expected_regret": float(task.p_high - info["reward_probability"][index]),
                            "mode0_probability": float(probs[index, 0]) if policy in ("sampled", "greedy") else None,
                            "score0": float(values[index, 0]) if policy in ("sampled", "greedy") else None,
                            "score1": float(values[index, 1]) if policy in ("sampled", "greedy") else None,
                            "mode_entropy": float(entropy[index]) if policy in ("sampled", "greedy") else None,
                            "same_context_omission": bool(
                                trial > 0 and previous_reward[index] == 0 and previous_high[index] == high
                            ),
                            "changed_arm": bool(trial > 0 and chosen != previous_arm[index]),
                        }
                    )
                z = torch.nn.functional.one_hot(torch.as_tensor(choices, device=device), num_classes=model.K).float()
                h = model.update_state(event, torch.as_tensor(reward, device=device), h, z)
                previous_arm = info["chosen_arm"]
                previous_reward = reward
                previous_high = info["high_arm"]
    finally:
        model.train(was_training)
    return summarize(rows, task, num_sessions), rows


def summarize(rows, task, num_sessions):
    def mean(field, subset=None):
        values = [r[field] for r in (rows if subset is None else subset) if r[field] is not None]
        return float(np.mean(values)) if values else None

    def curve(field, limit):
        result = []
        for lag in range(limit):
            subset = [r for r in rows if r[field] == lag]
            result.append(
                {
                    "lag": lag,
                    "n": len(subset),
                    "high_arm_accuracy": mean("high_arm_choice", subset),
                    "reward_rate": mean("reward", subset),
                }
            )
        return result

    # The stable window is explicitly the second half of nominal blocks;
    # actual-switch curves separately measure only experienced reversals.
    stable = [r for r in rows if r["trial_in_block"] >= task.block_length // 2]
    omissions = [r for r in rows if r["same_context_omission"]]
    sessions = []
    for session in range(num_sessions):
        subset = [r for r in rows if r["session"] == session]
        sessions.append(
            {
                "session": session,
                "high_arm_accuracy": mean("high_arm_choice", subset),
                "reward_rate": mean("reward", subset),
            }
        )
    return {
        "num_sessions": num_sessions,
        "completed_trials": len(rows),
        "actual_switches": sum(r["actual_switch"] for r in rows),
        "high_arm_accuracy": mean("high_arm_choice"),
        "reward_rate": mean("reward"),
        "expected_regret_per_trial": mean("expected_regret"),
        "stable_half_block_accuracy": mean("high_arm_choice", stable),
        "mode0_fraction": float(np.mean([r["mode"] == 0 for r in rows])),
        "mode_entropy": mean("mode_entropy"),
        "arm_change_after_omission_same_context": mean("changed_arm", omissions),
        "actual_switch_curve": curve("since_actual_switch", task.block_length),
        "nominal_block_curve": curve("trial_in_block", task.block_length),
        "sessions": sessions,
    }


def evaluation_suite(model, task, num_sessions, seed, output=None):
    result = {}
    for policy, reset in [
        ("sampled", False),
        ("greedy", False),
        ("greedy", True),
        ("random", False),
        ("win_stay_lose_shift", False),
        ("oracle", False),
    ]:
        name = policy + ("_memory_reset" if reset else "")
        summary, rows = evaluate(model, task, num_sessions, seed, policy, reset)
        result[name] = summary
        if output is not None:
            write_csv(output / f"{name}_trials.csv", rows)
    if output is not None:
        write_json(output / "evaluation.json", result)
    return result


def compact_evaluation(result):
    return {
        name: {k: value[k] for k in ("high_arm_accuracy", "reward_rate", "actual_switches")}
        for name, value in result.items()
    }


def train(args, model, task, output):
    history = TrialHistory(model.K, model.d_H, args.history_len)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    env = BlockSwitchBandit(task, args.num_envs, args.seed)
    event = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    updates = completed = 0
    started = time.monotonic()
    # Repeated validation has its own seed, distinct from final held-out test.
    validation_seed = args.seed + 1_000_000
    log_rows = []
    eval_rows = []
    with (output / "learning.jsonl").open("w") as log:
        for batch in range(1, args.session_batches + 1):
            env.reset()
            h = initial_hidden(model, args.num_envs)
            records = torch.zeros(args.num_envs, history.history_size, device=device)
            samples, accuracy, rewards = [], [], []
            losses, grad_norms = [], []
            for trial in range(env.session_trials):
                with torch.no_grad():
                    mode, indices = model.sample_mode(model.head(h))
                    reward_np, info = env.step(indices.cpu().numpy())
                    reward = torch.as_tensor(reward_np, device=device)
                    samples.append((records.clone(), h.clone(), indices.clone(), reward.clone()))
                    accuracy.extend((info["chosen_arm"] == info["high_arm"]).tolist())
                    rewards.extend(reward_np.tolist())
                    records = history.append_event(records, h, mode, reward, event)
                    h = model.update_state(event, reward, h, mode)
                completed += args.num_envs
                # Include every final outcome even when the update chunk is short.
                if len(samples) >= args.trials_per_update or env.done:
                    loss = trial_loss(model, history, samples)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite high-level loss at trial {completed}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    grad_norms.append(float(norm))
                    samples.clear()
                    updates += 1
            row = {
                "session_batch": batch,
                "completed_trials": completed,
                "optimizer_updates": updates,
                "high_arm_accuracy": float(np.mean(accuracy)),
                "reward_rate": float(np.mean(rewards)),
                "loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(grad_norms)),
                "elapsed_seconds": time.monotonic() - started,
            }
            log_rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if batch == 1 or batch % args.eval_every == 0 or batch == args.session_batches:
                evaluation = {}
                for policy in ("sampled", "greedy"):
                    evaluation[policy] = evaluate(model, task, args.eval_sessions, validation_seed, policy)[0]
                eval_rows.append(
                    {
                        "session_batch": batch,
                        "completed_trials": completed,
                        "optimizer_updates": updates,
                        "validation": evaluation,
                    }
                )
                print(json.dumps(dict(row, validation=compact_evaluation(evaluation))), flush=True)
                save_checkpoint(
                    output / f"manager_{completed:09d}.pt",
                    model,
                    args.history_len,
                    task.mode_to_arm,
                    extra={"completed_trials": completed, "optimizer_updates": updates},
                )
    write_csv(output / "learning.csv", log_rows)
    write_json(output / "validation.json", eval_rows)
    save_checkpoint(
        output / "manager.pt",
        model,
        args.history_len,
        task.mode_to_arm,
        extra={"completed_trials": completed, "optimizer_updates": updates},
    )
    return {"completed_trials": completed, "optimizer_updates": updates, "elapsed_seconds": time.monotonic() - started}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, required=True, help="New output directory (existing results are never overwritten)."
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--session-batches", type=int, default=500, help="Each batch runs num-envs independent complete sessions."
    )
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument(
        "--trials-per-update", type=int, default=4, help="Completed trials per environment, not maze frames."
    )
    p.add_argument("--history-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--loss", choices=("q", "policy"), default="q")
    p.add_argument(
        "--tau", type=float, default=1.0, help="Q sampling temperature; ignored by the existing policy model."
    )
    p.add_argument(
        "--eval-tau", type=float, help="Evaluation-only Q temperature override; does not modify saved weights/settings."
    )
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--adam-eps", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--block-length", type=int, default=15)
    p.add_argument("--blocks-per-session", type=int, default=5)
    p.add_argument("--switch-probability", type=float, default=0.66)
    p.add_argument("--p-high", type=float, default=0.8)
    p.add_argument("--p-low", type=float, default=0.2)
    p.add_argument("--initial-high-arm", choices=("left", "right", "random"), default="right")
    p.add_argument("--mode-to-arm", type=int, nargs=2, default=(0, 1), metavar=("MODE0_ARM", "MODE1_ARM"))
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-sessions", type=int, default=64)
    p.add_argument("--test-seed", type=int, default=2_000_007)
    p.add_argument("--init-checkpoint", type=Path, help="Weights-only initialization; CLI model settings must match.")
    p.add_argument("--evaluate-checkpoint", type=Path, help="Evaluation only; model settings come from the checkpoint.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in (
        "session_batches",
        "num_envs",
        "trials_per_update",
        "history_len",
        "hidden_size",
        "threads",
        "eval_every",
        "eval_sessions",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("tau", "learning_rate", "adam_eps", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.init_checkpoint and args.evaluate_checkpoint:
        raise ValueError("Choose either initialization or evaluation checkpoint")
    if args.eval_tau is not None and (not math.isfinite(args.eval_tau) or args.eval_tau <= 0):
        raise ValueError("eval-tau must be finite and positive")
    if args.test_seed in (args.seed, args.seed + 1_000_000):
        raise ValueError("test-seed must differ from training and validation seeds")
    task = BanditConfig(
        p_high=args.p_high,
        p_low=args.p_low,
        block_length=args.block_length,
        blocks_per_session=args.blocks_per_session,
        switch_probability=args.switch_probability,
        initial_high_arm=args.initial_high_arm,
        mode_to_arm=tuple(args.mode_to_arm),
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.evaluate_checkpoint:
        model, _ = build_model_from_checkpoint(
            args.evaluate_checkpoint, device=args.device, expected_mode_to_arm=task.mode_to_arm
        )
    else:
        model = HighLevelContextRNN_Learner(
            K=2, d_H=args.hidden_size, tau=args.tau, is_policy=args.loss == "policy"
        ).to(args.device)
        if args.init_checkpoint:
            metadata = load_checkpoint_into(model, args.init_checkpoint, expected_mode_to_arm=task.mode_to_arm)
            if metadata["history_len"] != args.history_len:
                raise ValueError("Initialization history-len must match checkpoint metadata")
    if model.K != 2 or model.representations:
        raise ValueError("Bandit requires K=2 and representations=False; decoder features are not synthesized")
    if args.eval_tau is not None and model.is_policy:
        raise ValueError("eval-tau only applies to the Q model; policy logits use no temperature")
    if (
        not args.evaluate_checkpoint
        and model.is_policy
        and args.num_envs * min(args.trials_per_update, task.session_trials) == 1
    ):
        raise ValueError("Policy loss needs multiple outcomes per update for its batch-mean reward baseline")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "config.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "task": asdict(task),
            "source": source_revision(),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "model": {
                "K": model.K,
                "d_H": model.d_H,
                "tau": model.tau,
                "is_policy": model.is_policy,
                "representations": model.representations,
            },
            "observation": "previous mode one-hot and binary outcome only; first trigger uses zero mode/reward",
            "actuator": "fixed bijection: each mode executes its configured arm with probability one",
            "optimization": "shared trial-history reconstruction and Q/policy loss; synchronous Adam; no PPO or PBT",
        },
    )
    accounting = None if args.evaluate_checkpoint else train(args, model, task, args.output)
    if args.eval_tau is not None:
        model.tau = args.eval_tau
    test = evaluation_suite(model, task, args.eval_sessions, args.test_seed, args.output)
    # Robustness changes task schedule, never the recurrent inputs or weights.
    generalization = {}
    for length in sorted({max(1, task.block_length // 2), task.block_length * 2}):
        shifted = replace(task, block_length=length, initial_high_arm="random")
        generalization[str(length)] = {
            policy: evaluate(model, shifted, args.eval_sessions, args.test_seed + 100 + length, policy)[0]
            for policy in ("sampled", "greedy")
        }
    write_json(args.output / "generalization.json", generalization)
    summary = {"training": accounting, "held_out": compact_evaluation(test)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
