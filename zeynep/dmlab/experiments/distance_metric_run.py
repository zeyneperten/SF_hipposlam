from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [11]),
    ]
)


vstr = "hipposlam"

cli = (
    "--env=openfield_map2_fixed_loc3 "
    "--seed=42 "
    "--train_for_seconds=108000 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--learning_rate=0.0002 "
    "--exploration_loss_coeff=0.005 "
    "--value_loss_coeff=0.3 "
    "--ppo_clip_ratio=0.25 "
    "--num_workers=32 "
    "--num_envs_per_worker=8 "
    "--worker_num_splits=8 "
    "--num_epochs=1 "
    "--rollout=64 "
    "--recurrence=64 "
    "--batch_size=2048 "
    "--num_batches_per_epoch=2 "
    "--decorrelate_experience_max_seconds=120 "
    "--max_grad_norm=0.0 "
    "--dmlab_renderer=software "
    "--dmlab_extended_action_set=False "
    "--dmlab_reduced_action_set=True "
    "--dmlab_one_task_per_worker=True "
    "--dmlab_use_level_cache=True "
    "--set_workers_cpu_affinity=False "
    "--num_policies=4 "
    "--with_pbt=True "
    "--pbt_replace_reward_gap=0.05 "
    "--pbt_replace_reward_gap_absolute=0.2 "
    "--pbt_period_env_steps=2000000 "
    "--pbt_start_mutation=10000000 "
    "--pbt_mix_policies_in_one_env=False "
    "--pbt_target_objective=lenweighted_score "
    "--pbt_perturb_max=1.3 "
    "--pbt_replace_fraction=0.2 "
    "--max_policy_lag=35 "
    "--use_record_episode_statistics=True "
    "--keep_checkpoints=10 "
    "--save_every_sec=120 "
    "--save_milestones_sec=5400 "
    "--save_best_every_sec=30 "
    "--decoder_mlp_layers 128 128 "
    "--env_frameskip=8 "
    "--core_name=BypassSS "
    "--DG_name=batchnorm_relu_fixed "
    "--DG_BN_intercept=2.43 "
    "--depth_sensor=True "
    "--normalize_input=False "
    "--fix_encoder_when_load=True "
    "--encoder_load_path=/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth "
    "--encoder_conv_architecture=pretrained_resnet "
    "--encoder_conv_mlp_layers=256 "
    "--use_rnn=True "
    "--rnn_type=gru "
    "--Hippo_n_feature=16 "
    "--Hippo_L=64 "
    "--rnn_size=1149 "
    "--nonlinearity=relu "
    "--with_wandb=False "
    "--wandb_user=xiaoxionglin-bernstein-center-freiburg "
    "--wandb_project=SF_DistanceMetric "
    "--benchmark=False "
    "--with_number_instruction=True "
    "--number_instruction_coef=9 "
    "--save_best_metric=avg_z_00_openfield_map2_fixed_loc3_lenweighted_score "
    "--device=cpu "
    "--train_dir=./train_dir"
    "--rec_distances=True"
    "--distance_learning=False"
    "--masked_distance_matrix=False"
    "--normalize_advantage=True"
)


_experiments = [
    Experiment("DistanceMetricLearning", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab.experiments.dmlab30
# Run on Slurm: python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab.experiments.dmlab30
# python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=train_dir/slurm_grid --slurm_gpus_per_job=0 --slurm_cpus_per_gpu=50 --slurm_sbatch_template=train_dir/training_templates/training_template.sh --pause_between=1 --slurm_print_only=False --run=sf_workingdir.dmlab.experiments.distance_metric_run --slurm_partition=genoa --slurm_timeout=0:10:00

# python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_grid --slurm_gpus_per_job=0 --slurm_cpus_per_gpu=32 --slurm_sbatch_template=./training_templates/training_template.sh --pause_between=1 --slurm_print_only=False --run=sf_workingdir.dmlab.experiments.run_test --slurm_partition=genoa --slurm_timeout=00:05:00
