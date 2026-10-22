wandb: [wandb.Api()] Loaded credentials for https://api.wandb.ai from /home/erten/.netrc.
Querying live W&B data across projects...

=== Processing Setup: No-Context ===
    W&B Project: xiaoxionglin-bernstein-center-freiburg/ymaze_nocontext
    Fetching runs list from xiaoxionglin-bernstein-center-freiburg/ymaze_nocontext...
    Total runs in ymaze_nocontext: 45
    Matched 5 seed run(s):
      - 05_noINSTR_see_1111_r.sca_0.1_l.rat_0.0002_20260916_144247_012118 (seed=1111)
      - 14_noINSTR_see_2222_r.sca_0.1_l.rat_0.0002_20260916_144254_826406 (seed=2222)
      - 23_noINSTR_see_3333_r.sca_0.1_l.rat_0.0002_20260916_144308_171133 (seed=3333)
      - 32_noINSTR_see_4444_r.sca_0.1_l.rat_0.0002_20260916_144326_667937 (seed=4444)
      - 41_noINSTR_see_5555_r.sca_0.1_l.rat_0.0002_20260916_144326_668611 (seed=5555)
    -> Aggregated No-Context: TL seeds=5, Flex seeds=5.

=== Processing Setup: DG: Sigmoid ===
    W&B Project: xiaoxionglin-bernstein-center-freiburg/ENC_DECymaze_norew_instr
    Fetching runs list from xiaoxionglin-bernstein-center-freiburg/ENC_DECymaze_norew_instr...
    Total runs in ENC_DECymaze_norew_instr: 100
    Matched 5 seed run(s):
      - 05_encoderonly_norew_INSTR_see_1111_l.rat_0.0001_D.c.mod_sigmoid_20260907_155912_319159 (seed=1111)
      - 11_encoderonly_norew_INSTR_see_2222_l.rat_0.0001_D.c.mod_sigmoid_20260907_160101_253288 (seed=2222)
      - 17_encoderonly_norew_INSTR_see_3333_l.rat_0.0001_D.c.mod_sigmoid_20260907_161000_695853 (seed=3333)
      - 23_encoderonly_norew_INSTR_see_4444_l.rat_0.0001_D.c.mod_sigmoid_20260907_161431_425680 (seed=4444)
      - 29_encoderonly_norew_INSTR_see_5555_l.rat_0.0001_D.c.mod_sigmoid_20260907_161942_497789 (seed=5555)

      - 02_encoderonly_norew_INSTR_see_1111_l.rat_2e-05_D.c.mod_sigmoid_20260907_155908_875328 (seed=1111)
      - 08_encoderonly_norew_INSTR_see_2222_l.rat_2e-05_D.c.mod_sigmoid_20260907_155919_680496 (seed=2222)
      - 14_encoderonly_norew_INSTR_see_3333_l.rat_2e-05_D.c.mod_sigmoid_20260907_160159_927031 (seed=3333)
      - 20_encoderonly_norew_INSTR_see_4444_l.rat_2e-05_D.c.mod_sigmoid_20260907_161320_772435 (seed=4444)
      - 26_encoderonly_norew_INSTR_see_5555_l.rat_2e-05_D.c.mod_sigmoid_20260907_161454_787947 (seed=5555)
    -> Aggregated DG: Sigmoid: TL seeds=5, Flex seeds=5.
    train_dir = /work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_

=== Processing Setup: DG: Concat ===
    W&B Project: xiaoxionglin-bernstein-center-freiburg/ENC_DECymaze_norew_instr
    Matched 5 seed run(s):
      - 03_encoderonly_norew_INSTR_see_1111_l.rat_0.0001_D.c.mod_concat_20260907_155908_875187 (seed=1111)
      - 09_encoderonly_norew_INSTR_see_2222_l.rat_0.0001_D.c.mod_concat_20260907_155919_680430 (seed=2222)
      - 15_encoderonly_norew_INSTR_see_3333_l.rat_0.0001_D.c.mod_concat_20260907_160159_927047 (seed=3333)
      - 21_encoderonly_norew_INSTR_see_4444_l.rat_0.0001_D.c.mod_concat_20260907_161320_772442 (seed=4444)
      - 27_encoderonly_norew_INSTR_see_5555_l.rat_0.0001_D.c.mod_concat_20260907_161454_784632 (seed=5555)

      - 00_encoderonly_norew_INSTR_see_1111_l.rat_2e-05_D.c.mod_concat_20260907_155852_287247 (seed=1111)
      - 06_encoderonly_norew_INSTR_see_2222_l.rat_2e-05_D.c.mod_concat_20260907_155912_318975 (seed=2222)
      - 12_encoderonly_norew_INSTR_see_3333_l.rat_2e-05_D.c.mod_concat_20260907_160101_250198 (seed=3333)
      - 18_encoderonly_norew_INSTR_see_4444_l.rat_2e-05_D.c.mod_concat_20260907_161000_695868 (seed=4444)
      - 24_encoderonly_norew_INSTR_see_5555_l.rat_2e-05_D.c.mod_concat_20260907_161431_425658 (seed=5555)
    -> Aggregated DG: Concat: TL seeds=5, Flex seeds=5.
    trian_dir = /work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_

=== Processing Setup: Decoder + DG: Concat+Sigmoid ===
    W&B Project: xiaoxionglin-bernstein-center-freiburg/NEWymaze_norew_instr
    Fetching runs list from xiaoxionglin-bernstein-center-freiburg/NEWymaze_norew_instr...
    Total runs in NEWymaze_norew_instr: 180
    Matched 5 seed run(s):
      - 13_sigmoid_norew_INSTR_see_1111_n.i.coe_200_r.sca_0.1_l.rat_0.0001_20260822_094245_925971 (seed=1111)
      - 31_sigmoid_norew_INSTR_see_2222_n.i.coe_200_r.sca_0.1_l.rat_0.0001_20260822_094306_102549 (seed=2222)
      - 49_sigmoid_norew_INSTR_see_3333_n.i.coe_200_r.sca_0.1_l.rat_0.0001_20260822_142211_235838 (seed=3333)
      - 67_sigmoid_norew_INSTR_see_4444_n.i.coe_200_r.sca_0.1_l.rat_0.0001_20260822_204500_870833 (seed=4444)
      - 85_sigmoid_norew_INSTR_see_5555_n.i.coe_200_r.sca_0.1_l.rat_0.0001_20260823_155030_223137 (seed=5555)

      - 12_sigmoid_norew_INSTR_see_1111_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_094245_926033 (seed=1111)
      - 30_sigmoid_norew_INSTR_see_2222_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_094306_102492 (seed=2222)
      - 48_sigmoid_norew_INSTR_see_3333_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_141651_075658 (seed=3333)
      - 66_sigmoid_norew_INSTR_see_4444_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_204500_870830 (seed=4444)
      - 84_sigmoid_norew_INSTR_see_5555_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260823_155030_501136 (seed=5555)
    -> Aggregated Decoder + DG: Concat+Sigmoid: TL seeds=5, Flex seeds=5.
    train_dir = /work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_

=== Processing Setup: Decoder ===
    W&B Project: xiaoxionglin-bernstein-center-freiburg/ENC_DECymaze_norew_instr
    Matched 5 seed run(s):
      - 01_decoderonly_norew_INSTR_see_1111_l.rat_0.0001_20260907_162216_665701 (seed=1111)
      - 03_decoderonly_norew_INSTR_see_2222_l.rat_0.0001_20260907_162615_918490 (seed=2222)
      - 05_decoderonly_norew_INSTR_see_3333_l.rat_0.0001_20260907_175351_994361 (seed=3333)
      - 07_decoderonly_norew_INSTR_see_4444_l.rat_0.0001_20260907_181558_152233 (seed=4444)
      - 09_decoderonly_norew_INSTR_see_5555_l.rat_0.0001_20260907_183146_803896 (seed=5555)

      - 00_decoderonly_norew_INSTR_see_1111_l.rat_2e-05_20260907_161942_499591 (seed=1111)
      - 02_decoderonly_norew_INSTR_see_2222_l.rat_2e-05_20260907_162304_426501 (seed=2222)
      - 04_decoderonly_norew_INSTR_see_3333_l.rat_2e-05_20260907_175351_994268 (seed=3333)
      - 06_decoderonly_norew_INSTR_see_4444_l.rat_2e-05_20260907_175351_994273 (seed=4444)
      - 08_decoderonly_norew_INSTR_see_5555_l.rat_2e-05_20260907_183146_803859 (seed=5555)
    -> Aggregated Decoder: TL seeds=5, Flex seeds=5.
    train_dir = /work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_