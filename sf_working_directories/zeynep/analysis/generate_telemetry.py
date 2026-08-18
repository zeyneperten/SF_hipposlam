# ---------------------------------------------------------------------------
# logging helpers (put near the top of the file, after imports)
# ---------------------------------------------------------------------------
import datetime
import json
import os
import pathlib
import sys
import time
from collections import deque
from glob import glob
from typing import Dict, Tuple

import gymnasium as gym
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import BaseLearner, create_learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log

# from sample_factory.enjoy import enjoy
from sf_working_directories.jannek.dmlab.enjoy_hipposlam import enjoy
from sf_working_directories.jannek.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components


def _ensure_parent(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)


mapname = "openfield_map2_fixed_loc3_noreward"
expname = "21_InternalRewardSeparateReward2_see_8_e.g.coe_1_e.r.met_punish"
train_dir = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam/InternalRewardSeparateReward2_"

cli = [
    "--algo",
    "APPO",
    "--env",
    mapname,  # pick any DM‑Lab level you have
    "--experiment",
    expname,
    "--encoder_load_path",
    "/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth",
    "--train_dir",
    train_dir,  # anything writable
    "--max_num_frames",
    "50000",  # short rollout for the test
    "--num_envs",
    "8",
    "--dmlab_level_cache_path",
    "./.dmlab_cache",
    "--load_checkpoint_kind",
    "latest",
    "--use_jit",
    "False",
    "--with_pos_obs",
    "True",
    "--no_render",  # <-- skip human window; avoid X11 on servers
]

cli_dict = {
    "algo": "APPO",
    "env": mapname,
    "experiment": expname,
    "encoder_load_path": "/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth",
    "train_dir": train_dir,
    "max_num_frames": "50000",
    "num_envs": "8",
    "dmlab_level_cache_path": "./.dmlab_cache",
    "load_checkpoint_kind": "latest",
    "no_render": True,
    "use_jit": False,
    "with_pos_obs": True,
}
register_dmlab_components()
cfg = parse_dmlab_args(evaluation=True, argv=cli)

# tweak whatever you like *after* parsing
# cfg.with_pos_obs = True
cfg.cli_args = cli_dict
# status = enjoy(cfg)


## Single enjoy run


verbose = False

cfg = load_from_checkpoint(cfg)

eval_env_frameskip: int = cfg.env_frameskip
assert cfg.env_frameskip % eval_env_frameskip == 0, f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

cfg.num_envs = 1

# render_mode = "human"
# if cfg.save_video:
#     render_mode = "rgb_array"
# elif cfg.no_render:
render_mode = None

env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode)
env_info = extract_env_info(env, cfg)

if hasattr(env.unwrapped, "reset_on_init"):
    # reset call ruins the demo recording for VizDoom
    env.unwrapped.reset_on_init = False
log.info(env.action_space)
actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
actor_critic.eval()


device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
actor_critic.model_to_device(device)

# learner = create_learner(cfg,env_info,)


#################### register hook
layers_to_log = [
    "encoder.basic_encoder.mlp_layers.0",
    "encoder.DG_projection.linear",
    "core",
    # "decoder.mlp.0",
    # "decoder.mlp.2"
]  # <- example; edit to taste

# 2B. activation buffer

import collections

act_buffers = collections.defaultdict(list)


def make_hook(layer_name):
    def _hook(_m, _inp, out):
        if isinstance(out, (tuple, list)):  # RNN returns (output, h_n)
            out = out[0]
        act_buffers[layer_name].append(out.detach().cpu())

    return _hook


# attach the hook once
for layer_to_log in layers_to_log:
    try:
        dict(actor_critic.named_modules())[layer_to_log].register_forward_hook(make_hook(layer_to_log))
        log.info("Activation hook registered on %s", layer_to_log)
    except KeyError:
        raise RuntimeError(f"Layer '{layer_to_log}' not found in the network!")

####################


policy_id = cfg.policy_index
# log.info(policy_id)
name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
# log.info(Learner.checkpoint_dir(cfg, policy_id))
checkpoints = BaseLearner.get_checkpoints(BaseLearner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
checkpoint_dict = BaseLearner.load_checkpoint(checkpoints, device)
actor_critic.load_state_dict(checkpoint_dict["model"])

episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
num_frames = 0

last_render_start = time.time()


def max_frames_reached(frames):
    return cfg.max_num_frames is not None and frames > cfg.max_num_frames


reward_list = []

obs, infos = env.reset()
rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
episode_reward = None
finished_episode = [False for _ in range(env.num_agents)]

video_frames = []
num_episodes = 0
num_traj = 0

# saved_data=dict()
pose_records = []

with torch.no_grad():
    while not max_frames_reached(num_frames):
        # log.info(f'Done Frames: {num_frames}')
        # log.debug(f'Generating Actions from Observation')
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

        # if not cfg.no_render:
        #     visualize_policy_inputs(normalized_obs)
        # log.debug(f'Generating Policy Outputs')
        policy_outputs = actor_critic(normalized_obs, rnn_states)
        # log.debug(f'Finished generating policy outputs')

        # sample actions from the distribution by default
        actions = policy_outputs["actions"]

        if cfg.eval_deterministic:
            # log.debug(f'Generating deterministic action distribution')
            action_distribution = actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

        # actions shape should be [num_agents, num_actions] even if it's [1, 1]
        # log.debug(f'Preprocessing actions')
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)

        rnn_states = policy_outputs["new_rnn_states"]

        for _ in range(render_action_repeat):
            # last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)
            # log.debug(f'Environment step')
            obs, rew, terminated, truncated, infos = env.step(actions)
            # log.info(obs['DEBUG.POS.TRANS'])
            # log.info(terminated)
            # save info
            frame_idx = num_frames  # or use a wall‑clock timestamp
            pos = obs["DEBUG.POS.TRANS"]  # (B,3)
            rot = obs["DEBUG.POS.ROT"]  # (B,3) or (B,4) depending on env
            # log.debug(f'Writing Information')
            for agent_i in range(env.num_agents):
                pose_records.append(
                    {
                        "frame": frame_idx,
                        "agent": agent_i,
                        "x": float(pos[agent_i, 0]),
                        "y": float(pos[agent_i, 1]),
                        "z": float(pos[agent_i, 2]),
                        "rot_x": float(rot[agent_i, 0]),
                        "rot_y": float(rot[agent_i, 1]),
                        "rot_z": float(rot[agent_i, 2]),
                        "num_traj": num_traj,
                        # keep the whole info dict as a JSON string for convenience
                        "info": json.dumps(infos[agent_i], default=str),
                    }
                )

            dones = make_dones(terminated, truncated)
            # log.info(dones)
            infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            num_frames += 1
            if num_frames % 100 == 0:
                log.debug(f"Num frames {num_frames}...")
            # log.debug(f'Checking for donsoes')
            dones = dones.cpu().numpy()
            for agent_i, done_flag in enumerate(dones):
                if done_flag:
                    num_traj += 1
                    log.info(done_flag)
                    log.info(cfg.use_record_episode_statistics)
                    finished_episode[agent_i] = True
                    rew = episode_reward[agent_i].item()
                    episode_rewards[agent_i].append(rew)

                    true_objective = rew
                    if isinstance(infos, (list, tuple)):
                        true_objective = infos[agent_i].get("true_objective", rew)
                    true_objectives[agent_i].append(true_objective)

                    if verbose:
                        log.info(
                            "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                            agent_i,
                            num_frames,
                            episode_reward[agent_i],
                            true_objectives[agent_i][-1],
                        )
                    rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    episode_reward[agent_i] = 0

                    if cfg.use_record_episode_statistics:
                        # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                        if "episode" in infos[agent_i].keys():
                            num_episodes += 1
                            reward_list.append(infos[agent_i]["episode"]["r"])
                    else:
                        num_episodes += 1
                        reward_list.append(true_objective)

            # if episode terminated synchronously for all agents, pause a bit before starting a new one
            if all(dones):
                # render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                time.sleep(0.05)
            # log.debug(f'Checking finished all.')
            if all(finished_episode):
                finished_episode = [False] * env.num_agents
                avg_episode_rewards_str, avg_true_objective_str = "", ""
                for agent_i in range(env.num_agents):
                    avg_rew = np.mean(episode_rewards[agent_i])
                    avg_true_obj = np.mean(true_objectives[agent_i])

                    if not np.isnan(avg_rew):
                        if avg_episode_rewards_str:
                            avg_episode_rewards_str += ", "
                        avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                    if not np.isnan(avg_true_obj):
                        if avg_true_objective_str:
                            avg_true_objective_str += ", "
                        avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                log.info("Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str)
                log.info(
                    "Avg episode reward: %.3f, avg true_objective: %.3f",
                    np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                    np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                )

            # VizDoom multiplayer stuff
            # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
            #     key = f'PLAYER{player}_FRAGCOUNT'
            #     if key in infos[0]:
            #         log.debug('Score for player %d: %r', player, infos[0][key])
        # log.info(num_episodes)
        if num_episodes >= cfg.max_num_episodes:
            break

env.close()


ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
telemetry = pathlib.Path(experiment_dir(cfg=cfg)) / "telemetry"
_ensure_parent(telemetry)
pose_path = telemetry / f"pose_{ts}.parquet"

csv_path = pose_path.with_suffix(".csv")
_ensure_parent(csv_path)
pddata = pd.DataFrame(pose_records)
pddata.to_csv(csv_path, index=False)
log.info("Saved %d pose rows to %s", len(pose_records), csv_path)

# log.info("Saved %d pose rows to %s", len(pose_records), pose_path)

# 5B.  save activations to HDF5  ----------------------------------------
act_path = telemetry / f"activations_{ts}.h5"
with h5py.File(act_path, "w") as h5:
    for layer, lst in act_buffers.items():
        if not lst:  # nothing recorded for that layer
            continue
        data = torch.cat(lst, dim=0).numpy()  # (frames*agents, …)
        h5.create_dataset(layer, data=data, compression="gzip")
        log.info("Saved %-20s  shape=%r", layer, data.shape)
