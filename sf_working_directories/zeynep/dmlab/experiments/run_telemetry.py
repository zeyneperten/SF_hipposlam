from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

# List the experiment names you want telemetry for
_params = ParamGrid(
    [
        (
            "experiment",
            [
                "00_Q_noclassifier_logperK_gradclip_see_1111_D.c.mod_additive_c.i.coe_1",
                "01_Q_noclassifier_logperK_gradclip_see_2222_D.c.mod_additive_c.i.coe_1",
                "02_Q_noclassifier_logperK_gradclip_see_3333_D.c.mod_additive_c.i.coe_1",
            ],
        ),
    ]
)

train_dir = "/work/classic/fr_ze12-data/ymaze_newRNN/HighLevelChoice/stage2_rollout/noclassifier/run2_add_logperK_gradclip/train_dir/Q_noclassifier_logperK_gradclip/Q_noclassifier_logperK_gradclip_"

cli = (
    "--algo=APPO "
    "--env=ymaze_instr_hl "
    f"--train_dir={train_dir} "
    "--max_num_frames=20000 "
    "--num_envs=8 "
    "--use_jit=False "
    "--with_pos_obs=True "
    "--no_render "
    "--core_name=BypassSS_HighLevelRNN "
    "--rnn_size=1342 "
    "--Decoder_context_mod=additive "
    "--hl_tau=1.0 "
    "--hl_history_len=8 "
)

_experiments = [
    Experiment("telemetry", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription("telemetry", experiments=_experiments)
