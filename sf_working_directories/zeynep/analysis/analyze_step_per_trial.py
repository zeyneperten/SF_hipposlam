import pandas as pd

df_trials = pd.read_csv("/work/classic/fr_ze12-data2/ymaze_CTRL/08_decoderonly_norew_INSTR_see_5555_l.rat_2e-05/trial_by_trial_choices.csv")

# 1. Frames per trial
frames_per_trial = (df_trials["frame"].iloc[-1] - df_trials["frame"].iloc[0]) / len(df_trials)

# 2. Decision steps per trial (frameskip = 8)
frameskip = 8
steps_per_trial = frames_per_trial / frameskip

# 3. Decision steps in a 15-trial block
steps_per_block = steps_per_trial * 15

print(f"Average frames per trial: {frames_per_trial:.1f}")
print(f"Agent decision steps per trial: {steps_per_trial:.1f}")
print(f"Agent decision steps per 15-trial block: {steps_per_block:.1f}")