import argparse
import os
from os.path import join

from sample_factory.utils.utils import str2bool


def hipposlam_override_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        encoder_conv_architecture="convnet_impala",
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        nonlinearity="relu",
        rollout=32,
        recurrence=32,
        rnn_type="lstm",
        rnn_size=256,
        num_epochs=1,
        # if observation normalization is used, it is important that we do not normalize INSTRUCTIONS observation
        normalize_input_keys=["obs"],
        decoder_mlp_layers=[128, 128],
    )


def add_hipposlam_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser
    p.add_argument("--Hippo_n_feature", default=64, type=int, help="number of sequences/features")
    p.add_argument("--Hippo_R", default=8, type=int, help="number of repeats in a sequence")
    p.add_argument("--Hippo_L", default=48, type=int, help="sequence length")

    p.add_argument(
        "--simple_sequence",
        default=False,
        type=bool,
        help="simple sequence, simply shrinking feature dimensions and expanding features to include their history",
    )
    p.add_argument(
        "--core_name",
        default=None,
        type=str,
        help="simple sequence, simply shrinking feature dimensions and expanding features to include their history",
    )
    p.add_argument("--encoder_name", default=None, type=str, help="actually using dmlab encoders")
    p.add_argument("--encoder_load_path", default=None, type=str, help="if loading encoder, the path")

    p.add_argument("--DG_lr", default=None, type=float, help="Dentate Gyrus Pattern separation learning rate")
    p.add_argument("--DG_temperature", default=None, type=float, help="Dentate Gyrus output temperature")
    p.add_argument(
        "--DG_batch_q", default=None, type=bool, help="Dentate Gyrus batch quantile, momentum 0.2, quantile 0.98"
    )
    p.add_argument("--DG_softmax", default=None, type=bool, help="Dentate Gyrus softmax")
    p.add_argument(
        "--DG_name", default=None, type=str, help="model name for the last encoder layer, i.e. Dentate Gyrus"
    )
    p.add_argument(
        "--DG_detect", default=None, type=float, help="batch novelty detection threshold (to activate a sequence)"
    )
    p.add_argument(
        "--DG_novelty", default=None, type=float, help="batch novelty novelty threshold to store a new pattern"
    )
    # p.add_argument("--dense", default=None, type=bool, help="whether encoder gives additional dense output")
    p.add_argument("--head_l1_coef", default=None, type=float, help="L1 penalty to encoder output")
    p.add_argument(
        "--fix_encoder_when_load",
        default=True,
        type=bool,
        help="when loading an encoder, fix its weights at initialization",
    )
    p.add_argument("--depth_sensor", default=False, type=bool, help="having extra depth sensor")
    p.add_argument(
        "--dmlab_reduced_action_set", default=False, type=bool, help="reduced action set to facilitate learning"
    )
    p.add_argument(
        "--with_number_instruction", default=True, type=str2bool, help="instruction input is number, e.g. 1-3"
    )
    p.add_argument("--number_instruction_coef", default=1, type=float, help="instruction strength")
    p.add_argument("--DG_BN_intercept", default=2, type=float, help="instruction strength")
    p.add_argument("--with_pos_obs", default=False, type=str2bool, help="get the true position of agent")
    p.add_argument(
        "--use_jit",
        default=True,
        type=str2bool,
        help="use jit / pytorch script to accelerate decoder. disable it for hooking",
    )

    p.add_argument(
        "--refractory",
        default=0,
        type=int,
        help="when using bypassSS_binary, determine whether to block reentry and how much the refractory. 0: no refractory, -1: entire sequence",
    )

    # p.add_argument("--rec_distances", default=None, type=bool, help="Record the distance between the propagation of each individual sequence")

    p.add_argument("--other_checkpint_path", default=None, type=str, help="load from other exps checkpoints")

    ###    Transformer decoder parameters

    p.add_argument("--load_model_path", default=None, type=str, help="Path to specific .pth file for the entire model")

    p.add_argument(
        "--reset_critic",
        default=False,
        type=str2bool,
        help="Reset all critic parameters. Useful after pre-training.",
    )
    p.add_argument(
        "--reset_decoder",
        default=False,
        type=str2bool,
        help="Reset all Decoder parameters (This includes Action-Parametrization/Value-Estimator Layers as well). Useful after pre-training.",
    )
    p.add_argument(
        "--double_value",
        default=False,
        type=str2bool,
        help="Only used in Janneks experiments.",
    )

    p.add_argument(
        "--decoder_type",
        default="mlp",
        type=str,
        choices=["mlp", "sr_transformer"],
        help="Decoder type: standard MLP or shift-register transformer",
    )

    # p.add_argument(
    #     "--decoder_sr_T",
    #     default=None,
    #     type=int,
    #     help="Shift-register window length T (expanded_length). If None, inferred as Hippo_R + Hippo_L - 1.",
    # )
    # p.add_argument(
    #     "--decoder_sr_df",
    #     default=None,
    #     type=int,
    #     help="Per-token feature dimension d_f (usually Hippo_n_feature). If None, inferred from Hippo_n_feature.",
    # )
    # p.add_argument(
    #     "--decoder_sr_bypass_size",
    #     default=None,
    #     type=int,
    #     help="Bypass feature size appended after T*d_f. If None, inferred from core_input_size.",
    # )
    p.add_argument(
        "--decoder_sr_include_bypass_in_output",
        default=True,
        type=str2bool,
        help="If True, append bypass features to decoder output",
    )

    # -------------------------------------------------------------------------
    # Transformer core (decoder)
    # -------------------------------------------------------------------------
    p.add_argument("--decoder_attn_d_model", default=64, type=int, help="Transformer model width d_model")
    p.add_argument("--decoder_attn_n_heads", default=1, type=int, help="Number of attention heads")
    p.add_argument(
        "--decoder_attn_d_ff",
        default=None,
        type=int,
        help="Transformer FFN hidden size. If None, uses 4*d_model",
    )
    p.add_argument("--decoder_attn_dropout", default=0.0, type=float, help="Dropout probability in transformer decoder")

    # -------------------------------------------------------------------------
    # Positional encoding for decoder transformer
    # -------------------------------------------------------------------------
    p.add_argument(
        "--decoder_attn_pos_mode",
        default="rope",
        type=str,
        choices=["none", "sin_add", "learned_add", "concat_sin", "concat_fourier", "rope", "concat_smoothed"],
        help="Positional encoding mode for transformer decoder",
    )
    p.add_argument(
        "--decoder_attn_d_p",
        default=16,
        type=int,
        help="Positional encoding dimension when using concat_* modes",
    )
    p.add_argument(
        "--decoder_attn_fourier_max_freq",
        default=1.0,
        type=float,
        help="Max frequency for concat_fourier positional encoding",
    )
    p.add_argument(
        "--decoder_attn_rope_base",
        default=100.0,
        type=float,
        help="Base for RoPE positional encoding",
    )

    # -------------------------------------------------------------------------
    # Decoder output
    # -------------------------------------------------------------------------
    p.add_argument(
        "--decoder_attn_out_dim",
        default=128,
        type=int,
        help="Output dimension of transformer decoder (before policy head)",
    )

    # Smoothed canonical time basis (only used when pos_mode=concat_smoothed)
    p.add_argument(
        "--decoder_attn_time_basis_normalize",
        default=True,
        type=str2bool,
        help="Normalize rows of smoothed time basis so they sum to 1",
    )

    # Readout options
    p.add_argument("--decoder_attn_readout_mode", default="last", type=str, help="Readout mode: last|weighted_sum")
    p.add_argument(
        "--decoder_attn_readout_attn_hidden",
        default=0,
        type=int,
        help="Hidden dim for pooling scorer. 0 => linear scorer.",
    )
