"""Focused synthetic checks for the optional Y-maze diversity reward."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from sf_working_directories.zeynep.dmlab.high_level_diversity import (
    TrialEndModeClassifier,
    predictability_bonus,
    trial_end_bonus,
    trial_end_classifier_loss,
)


class ImageCodeClassifier(nn.Module):
    include_chosen_arm = False

    def forward(self, image, chosen_arm=None):
        code = image[:, 0, 0, 0].long()
        return F.one_hot(code, num_classes=2).float() * 8.0


class HighLevelDiversityTests(unittest.TestCase):
    def test_constant_prediction_and_one_mode_get_no_bonus(self):
        logits = torch.tensor([[3.0, -3.0]] * 4)
        modes = torch.tensor([0, 0, 0, 1])
        self.assertTrue(torch.equal(predictability_bonus(logits, modes), torch.zeros(4)))
        self.assertTrue(torch.equal(predictability_bonus(logits, torch.zeros(4, dtype=torch.long)), torch.zeros(4)))

    def test_chance_level_guesses_have_no_positive_average_reward(self):
        modes = torch.tensor([0, 1, 0, 1])
        guesses = torch.tensor([0, 0, 1, 1])
        logits = F.one_hot(guesses, num_classes=2).float() * 8.0
        reward = predictability_bonus(logits, modes)
        self.assertEqual((guesses == modes).float().mean().item(), 0.5)
        self.assertLessEqual(reward.mean().item(), 0.0)
        self.assertGreater(reward.mean().item(), -0.01)
        self.assertTrue(torch.all(reward.abs() <= 1.0))

    def test_reward_uses_reached_state_and_previous_action_mode(self):
        # Two trajectories, three actions, and four observations each.
        obs = {
            "obs": torch.zeros(2, 4, 1, 1, 1),
            "outcome_event": torch.zeros(2, 4, 1),
        }
        obs["outcome_event"][:, 3] = 1.0
        obs["obs"][1, 3] = 1.0
        modes = F.one_hot(torch.tensor([[0, 0, 0], [1, 1, 1]]), num_classes=2).float()
        dones = torch.zeros(2, 3, dtype=torch.bool)
        valids = torch.ones(2, 3, dtype=torch.bool)

        bonus = trial_end_bonus(ImageCodeClassifier(), obs, modes, dones, valids, 0.01)
        self.assertTrue(torch.equal(bonus[:, :2], torch.zeros(2, 2)))
        self.assertTrue(torch.all(bonus[:, 2] > 0))
        self.assertTrue(torch.all(bonus <= 0.01))

        dones[0, 2] = True
        valids[1, 2] = False
        self.assertEqual(trial_end_bonus(ImageCodeClassifier(), obs, modes, dones, valids, 0.01).sum().item(), 0.0)

    def test_classifier_loss_updates_classifier_only_on_outcomes(self):
        classifier = TrialEndModeClassifier(1, 2)
        obs = {
            "obs": torch.zeros(4, 1, 32, 32),
            "outcome_event": torch.ones(4, 1),
        }
        obs["obs"][2:] = 1.0
        modes = F.one_hot(torch.tensor([0, 0, 1, 1]), num_classes=2).float()
        valid = torch.tensor([True, True, True, False])
        loss, count, accuracy = trial_end_classifier_loss(classifier, obs, modes, valid)
        self.assertEqual(count, 3)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in classifier.parameters()))

        obs["outcome_event"].zero_()
        loss, count, _ = trial_end_classifier_loss(classifier, obs, modes, valid)
        self.assertEqual(count, 0)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
