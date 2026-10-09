"""Optional trial-end mode discriminator for Y-maze controller rewards."""

import torch
from torch import nn
from torch.nn import functional as F


class TrialEndModeClassifier(nn.Module):
    """Predict the previous mode from the VISUAL OBSERVATION reached at trial end.

    Only the visual observation is used by default. The one-hot chosen arm can
    be included for levels that return to an identical center view at outcome.
    Neither the selected mode nor the reward is an input to the classifier.

    The classifier tries to guess which mode was selected.
    """

    def __init__(self, image_channels: int, num_modes: int, include_chosen_arm: bool = False):
        super().__init__()
        if num_modes < 2:
            raise ValueError("The diversity reward requires at least two modes")
        self.num_modes = num_modes
        self.include_chosen_arm = include_chosen_arm # if True, the classifier will take the chosen arm as an additional input. 

        # Convolutional Neural Network (CNN) to process the visual input. The architecture consists of two convolutional layers followed by ReLU activations, 
        # an adaptive average pooling layer, and a flattening operation to prepare the features for the fully connected layers.
        self.visual = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # flat head to output the final guess for the mode. The input size is determined by the output of the visual CNN and whether the chosen arm is included.
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4 + (3 if include_chosen_arm else 0), 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, image: torch.Tensor, chosen_arm: torch.Tensor | None = None) -> torch.Tensor:
        features = self.visual(image.float())
        if self.include_chosen_arm:
            if chosen_arm is None:
                raise ValueError("chosen_arm is required when include_chosen_arm=True")
            arm = F.one_hot(chosen_arm.long().reshape(-1).clamp(0, 2), num_classes=3)
            features = torch.cat((features, arm.to(features.dtype)), dim=-1)
        return self.head(features)


def predictability_bonus(logits: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Bounded, signed evidence for the true mode above its batch prior.

    A classifier that emits the same probabilities for every reached state
    receives zero bonus, even if one mode is selected much more often. Keeping
    negative evidence prevents chance-level guesses from earning reward.
    """
    if logits.size(0) < 2 or modes.unique().numel() < 2: # First check if at least 2 modes are used in this batch, if lazy give 0 bonus. 
        return logits.new_zeros(modes.shape)
    probabilities = logits.softmax(dim=-1)
    true_probability = probabilities.gather(1, modes[:, None]).squeeze(1) # Classifier's confidence in the true mode for each example in the batch.
    marginal_probability = probabilities.mean(dim=0)[modes] # How often Classifier guesses the mode overall. If blindly guess the same mode -> 100% -> image give no clues
    evidence = true_probability.clamp_min(1e-8).log() - marginal_probability.clamp_min(1e-8).log() # "Information gain" If the image actually helped to better guess than average
    return (evidence / torch.log(logits.new_tensor(logits.size(-1)))).clamp(-1.0, 1.0)


@torch.no_grad()
def trial_end_bonus(
    classifier: TrialEndModeClassifier,
    normalized_obs,
    selected_modes: torch.Tensor,
    dones: torch.Tensor,
    valids: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Return a bonus for action t using observation t+1 at trial end.
       Scan through the memory buffer (64-frame rollout) and give the bonus at correct time step (t) when the trial ends (t+1)."""
    next_outcome = normalized_obs["outcome_event"][:, 1:].reshape_as(dones) > 0.5 # Check if the next observation indicates a trial outcome event (e.g., reaching the end of a trial).
    mask = next_outcome & ~dones & valids & (selected_modes.sum(dim=-1) > 0.5)
    bonus = dones.new_zeros(dones.shape, dtype=torch.float32)
    if mask.any():
        image = normalized_obs["obs"][:, 1:][mask]
        arm = normalized_obs["chosen_arm"][:, 1:][mask] if classifier.include_chosen_arm else None
        labels = selected_modes.argmax(dim=-1)[mask]
        bonus[mask] = coefficient * predictability_bonus(classifier(image, arm), labels) # Get a score 0-1 and multiply by coefficient to scale the bonus.
    return bonus # Bonus is only at the exact frame where the trial ends, and is zero otherwise.


def trial_end_classifier_loss(
    classifier: TrialEndModeClassifier,
    normalized_obs,
    pre_event_modes: torch.Tensor,
    valids: torch.Tensor,
):
    """Fit on valid outcome observations and the mode held before that event.
       Train the classifier. It has its own CNN, weights to be updated!"""
    outcome = normalized_obs["outcome_event"].reshape(-1) > 0.5 # valid trial ends in the batch.
    mask = outcome & valids & (pre_event_modes.sum(dim=-1) > 0.5)
    count = int(mask.sum().item())
    if count == 0:
        return pre_event_modes.new_zeros(()), 0, 0.0
    image = normalized_obs["obs"][mask]
    arm = normalized_obs["chosen_arm"][mask] if classifier.include_chosen_arm else None
    labels = pre_event_modes.argmax(dim=-1)[mask] # Get the actual modes the agent was in
    logits = classifier(image, arm) # Guess the modes based on images
    # Equalize represented modes so a majority-only predictor is not enough.
    counts = torch.bincount(labels, minlength=classifier.num_modes).clamp_min(1) # Count how many times a mode was picked.
    weights = counts[labels].float().reciprocal()
    per_example_loss = F.cross_entropy(logits, labels, reduction="none")
    loss = (per_example_loss * weights).sum() / weights.sum() # cross entropy loss (how wrong the classifir is)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item() # accuracy percentage
    return loss, count, accuracy
