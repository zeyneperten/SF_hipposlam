import glob
import itertools
import os

import torch
import torch.nn.functional as F


def check_weight_collapse(checkpoint_path=None, exp_dir=None):
    # 1. FIND THE CHECKPOINT (.pth file)
    if checkpoint_path is None and exp_dir is not None:
        # Search for the latest checkpoint in the folder
        checkpoints = glob.glob(os.path.join(exp_dir, "*.pth"))
        if not checkpoints:
            print(f"❌ No .pth files found in {exp_dir}")
            return

        # Pick the one with the highest modification time (latest save)
        checkpoint_path = max(checkpoints, key=os.path.getmtime)
        print(f"Found latest checkpoint: {checkpoint_path}")

    if checkpoint_path is None:
        print("❌ Please provide a checkpoint_path or exp_dir")
        return

    print(f"\nLoading raw trained weights directly from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # SF stores the state dict inside 'model', 'model_state_dict', or at root
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 2. LOCATE THE Z-MODULATION WEIGHTS
    z_weights = None
    target_name = None

    for key in state_dict.keys():
        if "film_gamma.weight" in key or "context_proj.weight" in key:
            target_name = key
            z_weights = state_dict[key].detach()
            break

    if z_weights is None:
        decoder_keys = [k for k in state_dict.keys() if "decoder.mlp" in k and "weight" in k]

        if decoder_keys:
            # Sort the keys alphabetically/numerically so we guarantee we grab
            # the VERY FIRST layer (e.g., 'decoder.mlp.0.weight')
            target_name = sorted(decoder_keys)[0]
            full_weight = state_dict[target_name].detach()

            # Because z is concatenated at the end of the visual features,
            # we slice exactly the last K columns!
            z_weights = full_weight[:, -4:]
    # Final Check
    if z_weights is None:
        print("❌ Could not find FiLM, Additive, or Concatenation weights in the checkpoint!")
        return

    print(f"✅ Extracted weights from: {target_name}")
    print(f"   Shape of extracted z_weights: {z_weights.shape}")

    # 3. COMPUTE METRICS
    # nn.Linear weights are shape [out_features, in_features]
    # We want columns to represent the modes (K)
    if z_weights.shape[0] < z_weights.shape[1]:
        # If rows are smaller than columns, it's likely transposed
        z_weights = z_weights.T

    K = z_weights.shape[1]
    if "film" in target_name:
        mod_type = "FiLM"
    elif "additive" in target_name:
        mod_type = "Additive"
    else:
        mod_type = "Concatenation"

    print(f"Exact raw sum of all weights: {z_weights.sum().item()}")

    print("\n" + "=" * 50)
    print(f"MODE COLLAPSE ANALYSIS ({mod_type.upper()})")
    print("=" * 50)

    # Check 1: L2 Norms
    norms = torch.norm(z_weights, p=2, dim=0)
    all_exactly_one = True
    for k in range(K):
        norm_val = norms[k].item()
        print(f"Mode {k} Weight Magnitude (L2 Norm): {norm_val:.6f}")
        if abs(norm_val - 1.0) > 1e-4:
            all_exactly_one = False

    print("-" * 50)

    # Check 2: Cosine Similarity
    for i, j in itertools.combinations(range(K), 2):
        vec_i = z_weights[:, i]
        vec_j = z_weights[:, j]

        cos_sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0), eps=1e-8).item()
        warning = " <--- ⚠️ COLLAPSED!" if cos_sim > 0.90 else ""
        print(f"Similarity between Mode {i} and Mode {j}: {cos_sim:.6f}{warning}")

    print("\n" + "=" * 50)
    if all_exactly_one:
        print("🚨 WARNING: The L2 norms are EXACTLY 1.000000!")
        print("🚨 This means PyTorch's `nn.init.orthogonal_` is untouched.")
        print("🚨 EITHER the checkpoint didn't save right, OR the loss gradient to this layer is 0.0.")
    else:
        print("🎉 SUCCESS: The norms are no longer 1.000000!")
        print("🎉 The model has successfully learned to move the weights during training!")


if __name__ == "__main__":
    # Specific checkpoint path provided
    checkpoint_path = "/work/classic/fr_ze12-data/ymaze_newRNN/HighLevelChoice/stage2_rollout/noclassifier/run3_concat/train_dir/Q_noclassifier_logperK_gradclip_concat/Q_noclassifier_logperK_gradclip_concat_/04_Q_noclassifier_logperK_gradclip_concat_see_5555/checkpoint_p2/checkpoint_000001684_13205504.pth"
    check_weight_collapse(checkpoint_path=checkpoint_path)
