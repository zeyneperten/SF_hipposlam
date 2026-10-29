from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        # -------------------------------------------------
        # Positional encoding sweep
        # -------------------------------------------------
        # Seeds
        ("seed", [1111, 2222, 3333, 4444, 5555]),
        # ("number_instruction_coef", [9, 200]),
        # ("reward_scale", [0.01, 0.1, 1.0]),
        # ("learning_rate", [0.00002, 0.0001]),
        # ("seed", [2222]),
        #("hl_tau", [0.2]),
        # ("hl_DG_mod", ["gated", "sigmoid", "multiply"]),
        # ("Decoder_context_mod", ["enter", "FiLM"]),
        # ("context_injection_coef", [1, 9, 200]),
    ]
)

# 01.10 - Bug fixed: q-loss addition to total loss function in learner
# 02.10 - Bug fixed: chosen_arm pulse removed, now chosen_arm is persistent until trial ends, might be
# the reason why HL-RNN is not learning to persist the mode (z) until trial ends. Now, the chosen_arm is persistent until trial ends, and the outcome_event_pulse is used to tick the HL-RNN only at outcome events.
# 02.10 - Policy gradient loss is now added to the total loss function in the learner, so that the HL-RNN can learn to output a policy (z) that maximizes the expected reward. The Q-loss is still used for the value function update, but it is not used for the policy update.
# 02.10 - The first run with persistent rewards in environment was buggy so I reverted back the reward pulse in envrionment but the others keep persistent.
# 05.10 - This configuration now uses Q regression and eight completed-trial
# records for high-level RNN training. See dmlab/HIGH_LEVEL_RNN_ROLLOUT_FIX.md.
# The optional controller diversity reward is explained in
# dmlab/HIGH_LEVEL_DIVERSITY_REWARD.md.

# _params = ParamGrid(
#     [
#         # ("seed", [ 5555]),
#         # (
#         #     ("Hippo_L","rnn_size"),
#         #     (

#         #         [ 64, 1149]

#         #     )
#         # ),
#         # ("num_envs_per_worker",[2,4,8]),
#         # ("batch_size",[1024,2048]),
#         # ("num_batches_per_epoch",[2,4]),

#         # ("env_frameskip",[8,4]),
#         ("seed", [0, 1111, 2222, 7777, 8888, 9999]),

#         # ("num_policies",[4,8,16]),
#         (
#             ("Hippo_L","rnn_size"),
#             (
#                 [ 1 , 141],
#                 [ 8 , 253],
#                 [ 16, 381],
#                 [ 32, 637],
#                 [ 64, 1149]

#             )
#         ),
#     ]
# )

# _params = ParamGrid(
#    [
#        ("seed", [1111, 2222, 3333, 4444, 5555]),
#        ("learning_rate", [0.00002, 0.0001, 0.0002]),
#        ("number_instruction_coef", [9, 200])
#    ]
# )

prj = "ymaze_HighLevelRNN_transfer"
vstr = "1_policy_mixedselection_sig_trigRew_2_23Oct"

cli = (
    "--env=ymaze_80 "
    f"--wandb_project={prj} "
    # "--seed=42 "
    "--train_for_seconds=144000 "
    "--algo=APPO "
    "--gamma=0.995 "
    "--gae_lambda=0.975 "
    "--use_rnn=True "
    "--num_workers=8 "
    "--num_envs_per_worker=4 "
    "--worker_num_splits=4 "
    "--num_epochs=1 "
    "--rollout=64 "
    "--recurrence=64 "
    "--batch_size=2048 "
    "--num_batches_per_epoch=2 "
    "--benchmark=False "
    "--max_grad_norm=0.0 "  # TRY 1 BECAUSE OF GRADIENT EXPLOSION WITH LORA, try default 0.0 for HL
    "--dmlab_renderer=software "
    "--decorrelate_experience_max_seconds=120 "
    "--nonlinearity=relu "
    "--rnn_type=gru "
    "--dmlab_extended_action_set=False "
    "--encoder_conv_architecture=layer2_resnet18 "
    "--encoder_conv_mlp_layers=256 "
    "--dmlab_one_task_per_worker=True "
    "--set_workers_cpu_affinity=False "
    "--dmlab_use_level_cache=True "
    "--num_policies=4 "
    "--pbt_replace_reward_gap=0.01 "
    "--pbt_replace_reward_gap_absolute=0.15 "
    "--pbt_period_env_steps=2000000 "
    "--pbt_start_mutation=10000000 "
    "--with_pbt=True "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=10 "
    "--save_every_sec=120 "
    "--save_milestones_sec=5400 "
    "--decoder_mlp_layers 64 64 "
    "--env_frameskip=4 "  # lowered from 8
    # "--dmlab_reduced_action_set=True "
    "--dmlab_navigation_action_set=True "
    "--core_name=BypassSS_HighLevelRNN "  # default was set to ByPassSS or try BypassSS_HighLevelRNN
    "--rnn_type=gru "
    "--DG_name=batchnorm_relu "
    "--learning_rate=0.0002 "  # default
    "--fix_encoder_when_load=True "
    # "--encoder_load_path=/home/fr/fr_xl1014/training/best_000025288_203030528_reward_94.185.pth "
    "--with_wandb=True "  # BE CAREFUL
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--pbt_mix_policies_in_one_env=False "
    "--pbt_target_objective=lenweighted_score "
    "--with_number_instruction=True "  # keep it true, oracle_context = False blocks instruction input to encoder
    "--save_best_metric=lenweighted_score "
    "--device=cpu "
    "--Hippo_n_feature=16 "
    "--number_instruction_coef=200 "
    "--DG_BN_intercept=2.43 "
    "--depth_sensor=True "
    "--normalize_input=False "
    "--Hippo_L=64 "
    "--Hippo_R=8 "
    "--rnn_size=1324 "  # 1166 base + 8 trial records * (16 pre-event h + 4 mode + reward + valid) = 1342
    # "--exploration_loss_coeff=0.005 "
    # "--value_loss_coeff=0.3 "
    # "--ppo_clip_ratio=0.25 "
    # "--pbt_perturb_max=1.3 "
    # "--pbt_replace_fraction=0.2 "
    "--save_best_every_sec=30 "
    # "--decoder_type=sr_transformer "
    "--reward_input=False "  # add + 1 to rnn size
    "--DG_context_mod=None "
    "--oracle_context=False "  # set to true if stage1 HL_RNN will use oracle to fix z
    "--Decoder_context_mod=enter "
    "--hl_K=2 "
    "--hl_d_H=16 " # if 64, rnn_size is 1756
    #"--hl_tau=1.0 "  # current behavior; sweep only after checking outcome-reward scale and mode exploration # TAU SET TO 1 GIVES 70% SUCESS, SO LOWER IT
    "--hl_history_len=8 "
    "--hl_is_policy=True "  # one-step Q regression on the mode that actually earned each trial reward
    # "--hl_diversity_reward_coef=0.01 " # at most 0.01 per completed trial, for controller PPO only
    # "--hl_diversity_include_chosen_arm=True " # include the chosen arm in the classifier input for levels that return to an identical center view at outcome
    "--reward_scale=0.1 "  # default 1 LOWERED BECAUSE OF TOO HIGH VALUE LOSS
    # "--context_injection_coef=200 "
    "--hl_DG_mod=sigmoid "
    "--oracle_freeze_controller=False "
    "--hl_mixed_selection=True "
    "--hl_dg_from_prev_z=True "
    "--oracle_init_checkpoint=/work/classic/fr_ze12-data2/ymaze_ablation/80_rew/run2_trigRew/train_dir/2_80_trigRew/2_80_trigRew_/00_2_80_trigRew_see_1111_D.c.mod_sigmoid_D.c.mod_enter/checkpoint_p0/checkpoint_000031560_255590400.pth "
)

_experiments = []


_experiments = [
    Experiment(vstr, cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab.experiments.dmlab30
# Run on Slurm: python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./template.sh --pause_between=1 --slurm_print_only=False --run=sf_xxl.dmlab.experiments.Hipposlam_batch_run_zeyneptrial
# python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_grid --slurm_gpus_per_job=0 --slurm_cpus_per_gpu=50 --slurm_sbatch_template=./template.sh --pause_between=1 --slurm_print_only=False --run=sf_working_directories.zeynep.dmlab.experiments.exp_ymaze_HLRNN_transfer --slurm_partition=genoa --slurm_timeout=40:10:00