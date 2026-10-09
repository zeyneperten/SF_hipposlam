"""Check that oracle and learned high-level modes reach decoder modulation."""

import torch
from torch.nn import functional as F

from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.zeynep.dmlab.custom_decoder import make_hipposlam_decoder


def test_high_level_decoder_uses_all_mode_dimensions_in_oracle_runs():
    for oracle in (False, True):
        for modulation in ("additive", "FiLM"):
            cfg = AttrDict(
                core_name="BypassSS_HighLevelRNN",
                oracle_context=oracle,
                hl_K=4,
                Decoder_context_mod=modulation,
                decoder_mlp_layers=[8],
                use_jit=False,
                nonlinearity="relu",
                context_injection_coef=1.0,
            )
            decoder = make_hipposlam_decoder(cfg, core_input_size=12)
            assert decoder.processed_core_size == 8
            assert decoder.context_dim == 4
            mode = F.one_hot(torch.tensor([0, 1]), num_classes=4).float()
            core_output = torch.cat((torch.zeros(2, 8), mode), dim=-1)
            assert torch.equal(core_output[:, decoder.processed_core_size:], mode)


def test_plain_decoder_keeps_mode_in_its_mlp_input():
    cfg = AttrDict(
        core_name="BypassSS_HighLevelRNN",
        oracle_context=False,
        hl_K=4,
        Decoder_context_mod="None",
        decoder_mlp_layers=[8],
        use_jit=False,
        nonlinearity="relu",
    )
    decoder = make_hipposlam_decoder(cfg, core_input_size=12)
    assert decoder.core_input_size == 12
