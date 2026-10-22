"""Standalone 75-trial high-level Q probe; no DeepMind Lab or low-level PPO.

Run from the repository root with PYTHONPATH=. python tests/probe_high_level_training.py.
Each episode here is eight parallel 75-trial environments for one policy.
"""

import argparse
from types import SimpleNamespace

import torch

from sf_working_directories.zeynep.dmlab.custom_core import HighLevelRNNWrapperCore
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import q_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--trials-per-update", type=int, default=4)
    parser.add_argument("--task", choices=("stationary", "latent"), default="stationary")
    parser.add_argument("--best-mode", type=int, choices=range(4), default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg = SimpleNamespace(
        Hippo_n_feature=2,
        Hippo_R=1,
        Hippo_L=2,
        hl_K=4,
        hl_d_H=16,
        hl_history_len=8,
        hl_is_policy=False,
        hl_deterministic=False,
        oracle_context=False,
    )
    core = HighLevelRNNWrapperCore(cfg, input_size=9)
    model = core.hl_learner
    model.tau = args.tau
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    num_envs, trials_per_episode = 8, 75
    updates = 0

    for episode in range(args.episodes):
        # Latent task: the rewarding mode is 0 or 1 and changes by episode;
        # no context cue is supplied. Stationary task: one mode always pays.
        if args.task == "latent":
            context = torch.tensor([(episode + env) % 2 for env in range(num_envs)])
        else:
            context = torch.full((num_envs,), args.best_mode, dtype=torch.long)
        hidden = torch.zeros(num_envs, 16)
        history = torch.zeros(num_envs, core.history_size)
        with torch.no_grad():
            mode, _ = model.sample_mode(model.head(hidden))
        collected = []
        successes = []

        for trial in range(trials_per_episode):
            with torch.no_grad():
                reward = (mode.argmax(-1) == context).float() * args.reward
                collected.append((history.clone(), hidden.clone(), mode.argmax(-1).clone(), reward.clone()))
                successes.append((reward > 0).float().mean().item())
                event = torch.ones(num_envs, dtype=torch.bool)
                history = core.append_event(history, hidden, mode, reward, event)
                hidden = model.update_state(event, reward, hidden, mode)
                mode, _ = model.sample_mode(model.head(hidden))

            if (trial + 1) % args.trials_per_update == 0 or trial == trials_per_episode - 1:
                histories = torch.cat([sample[0] for sample in collected])
                fallback_hidden = torch.cat([sample[1] for sample in collected])
                choices = torch.cat([sample[2] for sample in collected]).view(1, -1)
                targets = torch.cat([sample[3] for sample in collected]).view(1, -1)
                scores = core.scores_from_history(histories, fallback_hidden).unsqueeze(0)
                loss = q_loss(scores, choices, targets, torch.ones_like(targets, dtype=torch.bool))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                updates += 1
                collected.clear()

        if episode in (0, 1, 4, 9, 19, 49, 99, 199, 299, 399, args.episodes - 1):
            success = sum(successes[-25:]) / 25
            print(
                f"episode={episode + 1:3d} updates={updates:4d} "
                f"sampled_success_last25={success:.3f} q_loss={loss.item():.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
