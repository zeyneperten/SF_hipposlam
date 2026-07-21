# from asyncio.sslproto import add_flowcontrol_defaults
from email import header
from logging import warning

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor, device, nn

from sample_factory.model.encoder import ConvEncoder, Encoder, ResnetEncoder
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sf_working_directories.default.dmlab.dmlab30 import DMLAB_INSTRUCTIONS, DMLAB_VOCABULARY_SIZE
from sf_working_directories.default.dmlab.dmlab_model import DmlabEncoder


class DGProjectionBatchNovelty(nn.Module):
    def __init__(
        self,
        feature_dim,
        pattern_limit,
        detection_threshold=0.1,
        novelty_threshold=0.4,
        eps=1e-8,
        norm_coef=0.002,
        soft_gate_scale=10.0,
        bias=3.0,
    ):
        """
        Args:
          feature_dim (int): Dimension of the input features.
          pattern_limit (int): Fixed number of output units (stored patterns).
          detection_threshold (float): Threshold for activation based on the projection magnitude.
          novelty_threshold (float): Threshold for novelty detection based on singular values.
          eps (float): Small constant to avoid division by zero.
          norm_coef (float): Coefficient used in pattern replacement.
          soft_gate_scale (float): Scaling factor for the sigmoid.
          bias (float): Bias subtracted in the sigmoid so that at the detection threshold, the output is near zero.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pattern_limit = pattern_limit
        self.detection_threshold = detection_threshold
        self.novelty_threshold = novelty_threshold
        self.eps = eps
        self.norm_coef = norm_coef
        self.soft_gate_scale = soft_gate_scale
        self.bias = bias

        # Fixed buffer for stored patterns: shape (pattern_limit, feature_dim).
        self.register_buffer("patterns", torch.zeros(pattern_limit, feature_dim))
        # Boolean mask indicating which slots are active.
        self.register_buffer("pattern_mask", torch.zeros(pattern_limit, dtype=torch.bool))
        # Tensors to track cumulative activation counts and sample counts per slot.
        self.register_buffer("pattern_activation_counts", torch.zeros(pattern_limit))
        self.register_buffer("pattern_sample_counts", torch.zeros(pattern_limit))

    def forward(self, x):
        # Normalize input samples (differentiable operations).
        x_norm = x / (x.norm(dim=1, keepdim=True) + self.eps)
        batch_size = x_norm.size(0)

        # Update sample counts for active slots (buffer update outside of autograd).
        if self.pattern_mask.any():
            with torch.no_grad():
                self.pattern_sample_counts[self.pattern_mask] += batch_size

        # --- Novelty Detection ---
        if self.pattern_mask.any():
            active_patterns = self.patterns[self.pattern_mask]  # (n_active, feature_dim)
            proj = torch.matmul(x_norm, active_patterns.t())  # (batch_size, n_active)
            proj_span = torch.matmul(proj, active_patterns)  # (batch_size, feature_dim)
            x_null = x_norm - proj_span
        else:
            x_null = x_norm

        try:
            U, S, Vh = torch.linalg.svd(x_null, full_matrices=False)
            novel_mask = S > self.novelty_threshold
            k = int(novel_mask.sum().item())
            if k > 0:
                new_patterns = Vh[:k, :]
                new_patterns = F.normalize(new_patterns, p=2, dim=1)
                new_norms = S[:k] / batch_size
                with torch.no_grad():
                    self.update_patterns(new_patterns, new_norms, batch_size)
        except Exception as e:
            print("SVD failed:", e)

        # --- Differentiable Activation Computation ---
        # Compute similarity between normalized inputs and stored patterns.
        sim = torch.matmul(x_norm, self.patterns.t())  # (batch_size, pattern_limit)
        # Zero out similarities for inactive pattern slots.
        sim = sim * self.pattern_mask.to(sim.dtype)

        # Compute a differentiable activation for each pattern slot.
        # For each element, when sim is equal to detection_threshold,
        # the input to sigmoid becomes: soft_gate_scale*(0) - bias = -bias.
        # With bias=3.0, sigmoid(-3) is approximately 0.047, so near-zero.
        activations = torch.sigmoid(self.soft_gate_scale * (sim - self.detection_threshold) - self.bias)

        # Bookkeeping: update activation counts using a hard decision for pattern with highest similarity.
        with torch.no_grad():
            # We still identify a "most activated" pattern per sample for bookkeeping.
            _, max_indices = torch.max(sim, dim=1)
            # Use a simple threshold on the computed activations to decide if a sample is active.
            active_samples = activations.max(dim=1)[0] > self.detection_threshold
            if active_samples.any():
                upd = torch.bincount(max_indices[active_samples], minlength=self.pattern_limit).float()
                self.pattern_activation_counts += upd

        return activations

    def update_patterns(self, new_patterns, new_norms, current_batch_size):
        k_new = new_patterns.size(0)
        device = self.patterns.device

        # Compute normalized activation counts for stored patterns.
        stored_norm = self.pattern_activation_counts / (self.pattern_sample_counts + self.eps)
        stored_norm = stored_norm.to(device)

        stored_idx = torch.arange(self.pattern_limit, device=device)
        sorted_stored_idx = stored_idx[torch.argsort(stored_norm)]

        new_idx = torch.arange(k_new, device=new_patterns.device)
        sorted_new_idx = new_idx[torch.argsort(new_norms, descending=True)]

        replace_count = 0
        for new_i in sorted_new_idx:
            if replace_count < self.pattern_limit:
                candidate_stored_idx = sorted_stored_idx[replace_count]
                if new_norms[new_i] * self.norm_coef > stored_norm[candidate_stored_idx]:
                    self.patterns[candidate_stored_idx] = new_patterns[new_i]
                    self.pattern_activation_counts[candidate_stored_idx] = 0.0
                    self.pattern_sample_counts[candidate_stored_idx] = current_batch_size
                    self.pattern_mask[candidate_stored_idx] = True
                    replace_count += 1
        log.info(f"{replace_count} patterns replaced, batch size: {current_batch_size}")


class DGProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dg_lr: float = 0.001, weight_decay: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dg_lr = dg_lr
        self.weight_decay = weight_decay

        self.linear = nn.Linear(self.in_features, out_features, bias=False)
        self.activation = nn.ReLU()

        # Disable gradients for the linear layer as updates are custom.
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute raw output and apply activation.
        raw_output = self.linear(x)
        raw_output = self.activation(raw_output)

        # Create a "corrected" output: use a baseline of 0.1 for non-winning neurons,
        # and set the winning neuron to 1.0.
        with torch.no_grad():
            corrected_homeo = torch.full_like(raw_output, 0.1)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected_homeo.scatter_(1, max_idx, 1.0)

            corrected = torch.full_like(raw_output, 0)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected.scatter_(1, max_idx, 1.0)

        # Calculate the difference between the raw output and the corrected target.
        diff = raw_output - corrected_homeo

        # Compute the average weight update over the batch.
        batch_size = x.size(0)
        dW = torch.matmul(diff.t(), x) / batch_size

        # Update the weight matrix using the custom rule with weight decay.
        with torch.no_grad():
            # Incorporate weight decay directly into the update:
            self.linear.weight -= self.dg_lr * (dW + self.weight_decay * self.linear.weight)

        # Return the corrected output as the projection result.
        return corrected


class DGProjection_obsolete(nn.Module):
    def __init__(self, in_features: int, out_features: int, dg_lr: float = 0.001, weight_decay: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dg_lr = dg_lr
        self.weight_decay = weight_decay

        self.linear = nn.Linear(self.in_features, out_features, bias=False)
        self.activation = nn.ReLU()

        # Disable gradients for the linear layer as updates are custom.
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute raw output and apply activation.
        raw_output = self.linear(x)
        raw_output = self.activation(raw_output)

        # Create a "corrected" output: use a baseline of 0.1 for non-winning neurons,
        # and set the winning neuron to 1.0.
        with torch.no_grad():
            corrected_homeo = torch.full_like(raw_output, 0.1)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected_homeo.scatter_(1, max_idx, 1.0)

            corrected = torch.full_like(raw_output, 0)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected.scatter_(1, max_idx, 1.0)

        # Calculate the difference between the raw output and the corrected target.
        diff = raw_output - corrected_homeo

        # Compute the average weight update over the batch.
        batch_size = x.size(0)
        dW = torch.matmul(diff.t(), x) / batch_size

        # Update the weight matrix using the custom rule with weight decay.
        with torch.no_grad():
            # Incorporate weight decay directly into the update:
            self.linear.weight -= self.dg_lr * (dW + self.weight_decay * self.linear.weight)

        # Return the corrected output as the projection result.
        return corrected


class DGProjection_relu(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = self.activation(logits)

        return probs


class DGProjection_batchnorm_relu(nn.Module):
    def __init__(self, in_features: int, out_features: int, intercept=2):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.batchnorm1d = nn.BatchNorm1d(out_features, affine=False, momentum=0.1)
        self.activation = nn.ReLU()
        self.intercept = intercept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)  # Shape: [batch_size, out_features]
        x = self.batchnorm1d(x)
        # Replace logits with softmaxed probabilities.
        x = self.activation(x - self.intercept)

        return x


class DGProjection_batchnorm_relu_fixed(nn.Module):
    def __init__(self, in_features: int, out_features: int, intercept=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Freeze parameters of the linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

        self.batchnorm1d = nn.BatchNorm1d(out_features, affine=False, momentum=0.1)
        self.activation = nn.ReLU()
        self.intercept = intercept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)  # Shape: [batch_size, out_features] (fixed)
        x = self.batchnorm1d(x)
        x = self.activation(x - self.intercept)
        return x


class DGProjection_log_softmax(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = self.log_softmax(logits)

        return probs


class DGProjection_simple_softmax(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = F.softmax(logits, dim=1)

        return probs


class DGProjectionBatchSparsity(nn.Module):
    def __init__(self, in_features: int, out_features: int, active_percentage: float = 0.05):
        """
        Args:
            in_features: Number of input features.
            out_features: Number of output neurons.
            active_percentage: Desired percentage (per output neuron) of active samples in a batch.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.active_percentage = active_percentage
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For each output neuron, keep only the top-k activations across the batch,
        where k is calculated to be about active_percentage of the batch size.
        """
        logits = self.linear(x)  # shape: [batch_size, out_features]
        batch_size = logits.size(0)
        # Determine number of samples to activate per neuron (at least 1)
        k = max(1, int(self.active_percentage * batch_size))

        # Transpose to shape [out_features, batch_size] so we can work per neuron
        logits_t = logits.transpose(0, 1)  # shape: [out_features, batch_size]

        # For each output neuron, find the indices of the top-k samples
        _, indices = torch.topk(logits_t, k, dim=1)

        # Create a zero mask of the same shape as logits_t
        mask = torch.zeros_like(logits_t)
        # Scatter ones into the top-k positions for each neuron
        mask.scatter_(1, indices, 1.0)

        # Transpose the mask back to [batch_size, out_features]
        mask = mask.transpose(0, 1)

        # Use a straight-through estimator trick:
        # In the forward pass, the output is the hard mask (exactly 1% active),
        # while in the backward pass, gradients flow as if the operation were the identity.
        output = mask + logits - logits.detach()
        return output


class DGProjection_simple_top1(nn.Module):
    def __init__(self, in_features: int, out_features: int, temperature: float = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature

        # Standard linear layer with bias.
        self.linear = nn.Linear(self.in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute logits from the linear layer.
        logits = self.linear(x)

        # Use Gumbel Softmax with the 'hard' flag set to True.
        # This returns one-hot vectors during the forward pass, but maintains gradients via the soft approximation.
        output = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return output


class RunningActivationQuantile(nn.Module):
    def __init__(self, out_features, momentum=0.2, quantile=0.98):
        """
        Maintains a running estimate of the quantile (e.g., 98th percentile) for each neuron.

        Args:
            out_features (int): Number of output neurons.
            momentum (float): Weighting factor for new data.
            quantile (float): The desired quantile (e.g., 0.98 for the 98th percentile).
        """
        super().__init__()
        self.register_buffer("running_quantile", torch.zeros(out_features))
        self.momentum = momentum
        self.quantile = quantile

    def update(self, batch_activation):
        """
        Update the running quantile using the current batch's activations.

        Args:
            batch_activation (Tensor): Activations of shape [batch_size, out_features].
        """

        # Compute the quantile (e.g., 98th percentile) along the batch dimension.
        batch_q = torch.quantile(batch_activation.float(), q=self.quantile, dim=0)
        # Update the running quantile using an exponential moving average in a no-grad context.
        with torch.no_grad():
            updated = self.momentum * batch_q + (1 - self.momentum) * self.running_quantile
            self.running_quantile.copy_(updated)


class DGProjectionWithRunningQuantile(nn.Module):
    def __init__(self, in_features: int, out_features: int, momentum: float = 0.1, quantile: float = 0.97):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.running_quantile = RunningActivationQuantile(out_features, momentum=momentum, quantile=quantile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = F.softmax(logits, dim=1)

        # Update the running quantile using the current batch's probabilities.
        self.running_quantile.update(probs)

        # Get the current running threshold per neuron and broadcast to match batch size.
        threshold = self.running_quantile.running_quantile.unsqueeze(0)  # Shape: [1, out_features]

        # Create a binary mask: 1 if probability is above threshold, 0 otherwise.
        mask = (probs > threshold).float()

        # Use a straight-through estimator: forward pass uses the hard mask,
        # but gradients flow as if the operation were the identity.
        output = mask + probs - probs.detach()
        return output


##### Actual Encoders For This Project #####


class DepthEncoder(Encoder):
    def __init__(self, cfg, size=10):
        super().__init__(cfg)

        input_ch = 1
        log.debug("Num input channels for depth encoder: %d", input_ch)

        if cfg.encoder_conv_architecture == "resnet_impala" or cfg.encoder_conv_architecture == "pretrained_resnet":
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encoder_conv_architecture}")

        curr_input_channels = input_ch
        self.downsample = nn.Upsample(size=(1, 10))

        self.encoder_out_size = size

    def forward(self, obs: Tensor):
        x = self.downsample(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class FixedMobileNetSmallEncoder(Encoder):
    def __init__(self, cfg, obs_space, pretrained=True, fixed=True):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        # Load the pretrained MobileNetV3 Small weights from torchvision.
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # Freeze all parameters of MobileNet.
        if fixed:
            for param in mobilenet.parameters():
                param.requires_grad = False

            # Set the model to evaluation mode.
            mobilenet.eval()

        # Use the feature extractor (all layers up to the classifier)
        # Option 1: If you need a single feature vector, you can use the features and avgpool.
        self.features = mobilenet.features  # Feature extraction layers.
        self.avgpool = mobilenet.avgpool  # Global average pooling.

        self.encoder_out_size = 576
        # Optionally, if you need to add an extra projection layer,
        # uncomment the following line and adjust dimensions as needed.
        # self.projection = nn.Linear(576, desired_dim)

        self.model_to_device("cuda")

    def forward(self, x):
        # x should be a tensor of shape [N, 3, H, W] where H,W >= 224 (or resized accordingly).
        x = self.features(x)  # Pass through MobileNet features.
        x = self.avgpool(x)  # Global average pooling; output shape [N, 576, 1, 1].
        x = torch.flatten(x, 1)  # Flatten to shape [N, 576].

        # If using an extra projection layer, uncomment:
        # x = self.projection(x)

        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class HipposlamEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        if cfg.Hippo_n_feature:
            self.Hippo_n_feature = cfg.Hippo_n_feature
        else:
            self.Hippo_n_feature = 64
            log.info("hippo n feature not set, using default: {self.Hippo_n_feature}")
        self.depth_sensor = getattr(cfg, "depth_sensor", False)
        # self.depth_sensor=False
        log.info(self.depth_sensor)
        if self.depth_sensor:
            # obs_depth = obs_space["obs"]
            # obs_depth.shape[0]=1
            log.info(f"using depth sensor {self.depth_sensor}")
            self.depth_encoder = DepthEncoder(cfg, size=10)
        else:
            self.depth_sensor = False

        log.debug(f"original obs space: {obs_space['obs']}")
        obs_cnn = obs_space["obs"]

        obs_cnn = gym.spaces.Box(low=0, high=255, shape=(3, cfg.res_h, cfg.res_w), dtype=np.uint8)
        self.basic_encoder = make_img_encoder(cfg, obs_cnn)
        self.encoder_out_size = self.basic_encoder.get_out_size()

        self.with_number_instruction = cfg.with_number_instruction
        self.number_instruction_coef = getattr(cfg, "number_instruction_coef", 1)
        if self.with_number_instruction:
            # repurposed it to encode map number
            self.instructions_lstm_units = 3
        else:
            # same as IMPALA paper
            self.embedding_size = 20
            self.instructions_lstm_units = 64
            self.instructions_lstm_layers = 1

            padding_idx = 0
            self.word_embedding = nn.Embedding(
                num_embeddings=DMLAB_VOCABULARY_SIZE, embedding_dim=self.embedding_size, padding_idx=padding_idx
            )

            self.instructions_lstm = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.instructions_lstm_units,
                num_layers=self.instructions_lstm_layers,
                batch_first=True,
            )

        # learnable initial state?
        # initial_hidden_values = torch.normal(0, 1, size=(self.instructions_lstm_units, ))
        # self.lstm_h0 = nn.Parameter(initial_hidden_values, requires_grad=True)
        # self.lstm_c0 = nn.Parameter(initial_hidden_values, requires_grad=True)

        self.encoder_out_size += self.instructions_lstm_units
        log.info("DMLab policy head output size: %r", self.encoder_out_size)

        if cfg.DG_lr:
            self.dg_lr = getattr(cfg, "DG_lr", 0.001)

            self.DG_projection = DGProjection(self.encoder_out_size, cfg.Hippo_n_feature, self.dg_lr)
        elif cfg.DG_temperature:
            self.temperature = getattr(cfg, "DG_temperature", 1)
            self.DG_projection = DGProjection_simple_top1(self.encoder_out_size, cfg.Hippo_n_feature, self.temperature)
        elif cfg.DG_batch_q:
            self.DG_projection = DGProjectionWithRunningQuantile(self.encoder_out_size, cfg.Hippo_n_feature)
        elif cfg.DG_softmax:
            self.DG_projection = DGProjection_simple_softmax(self.encoder_out_size, cfg.Hippo_n_feature)
        else:
            self.DG_projection = nn.Linear(self.encoder_out_size, cfg.Hippo_n_feature)

        if cfg.DG_name == "log_softmax":
            self.DG_projection = DGProjection_log_softmax(self.encoder_out_size, cfg.Hippo_n_feature)
        elif cfg.DG_name == "linear_relu":
            self.DG_projection = DGProjection_relu(self.encoder_out_size, cfg.Hippo_n_feature)
        elif cfg.DG_name == "batch_novelty":
            self.dg_detect = getattr(cfg, "DG_detect", 0.1)
            if not self.dg_detect:
                print("getattr doesn't behave like you think, setting dg_detect to 0.1")
                self.dg_detect = 0.1
            self.dg_novelty = getattr(cfg, "DG_novelty", 0.4)
            if not self.dg_novelty:
                print("getattr doesn't behave like you think, setting dg_novelty to 0.4")
                self.dg_novelty = 0.4
            self.DG_projection = DGProjectionBatchNovelty(
                self.encoder_out_size, cfg.Hippo_n_feature, self.dg_detect, self.dg_novelty
            )
        elif cfg.DG_name == "batchnorm_relu":
            intercept = getattr(cfg, "DG_BN_intercept", 2)
            self.DG_projection = DGProjection_batchnorm_relu(
                self.encoder_out_size, cfg.Hippo_n_feature, intercept=intercept
            )
        elif cfg.DG_name == "batchnorm_relu_fixed":
            intercept = getattr(cfg, "DG_BN_intercept", 2)
            self.DG_projection = DGProjection_batchnorm_relu_fixed(
                self.encoder_out_size, cfg.Hippo_n_feature, intercept=intercept
            )

        tmp_out_size = cfg.Hippo_n_feature

        if cfg.core_name.startswith("SeqDense"):  # "Gate":
            self.n_dense_feature = getattr(cfg, "N_dense_feature", 16)
            if not self.n_dense_feature:
                print("getattr doesn't behave like you think, setting n_dense_feature to 16")
                self.n_dense_feature = 16
            self.dense = nn.Linear(self.encoder_out_size, self.n_dense_feature)
            tmp_out_size += self.n_dense_feature

        bypass_features = 0
        bypass_features = self.encoder_out_size
        if hasattr(cfg, "depth_sensor"):
            log.info(f"denpth_sensor {cfg.depth_sensor}")
            if self.depth_sensor:
                log.info(f"denpth_sensor {self.depth_sensor}")
                bypass_features = self.depth_encoder.get_out_size() + self.instructions_lstm_units

        self.bypass = False
        if cfg.core_name.startswith("Bypass"):  # "Gate":
            self.bypass = True
            tmp_out_size += bypass_features
            log.info(f"using bypass, dim {bypass_features}")

        self.encoder_out_size = tmp_out_size
        self.cpu_device = torch.device("cpu")

        # log.info("=================================== memory=========================")
        # log.info(torch.cuda.memory_allocated())

    def model_to_device(self, device):
        self.to(device)
        if self.with_number_instruction:
            return
        self.word_embedding.to(self.cpu_device)
        self.instructions_lstm.to(self.cpu_device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return self.cpu_device
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return torch.int64
        return torch.float32

    def forward(self, obs_dict):
        # obs_cnn = obs_dict["obs"].copy()
        if self.depth_sensor:
            obs_cnn = obs_dict["obs"][:, :3, :, :]
        else:
            obs_cnn = obs_dict["obs"][:, :, :, :]
        x = self.basic_encoder(obs_cnn)

        if self.with_number_instruction:
            instr = obs_dict[DMLAB_INSTRUCTIONS]
            last_outputs = (
                torch.nn.functional.one_hot(instr.squeeze(1) - 1, num_classes=3) * self.number_instruction_coef
            )

            # log.info(last_outputs)

        else:

            with torch.no_grad():
                instr = obs_dict[DMLAB_INSTRUCTIONS]
                instr_lengths = (instr != 0).sum(axis=1)
                instr_lengths = torch.clamp(instr_lengths, min=1)
                max_instr_len = torch.max(instr_lengths).item()
                instr = instr[:, :max_instr_len]

            instr_embed = self.word_embedding(instr)
            instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
                instr_embed,
                instr_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            rnn_output, _ = self.instructions_lstm(instr_packed)
            rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

            first_dim_idx = torch.arange(rnn_outputs.shape[0])
            last_output_idx = sequence_lengths - 1
            last_outputs = rnn_outputs[first_dim_idx, last_output_idx]

        last_outputs = last_outputs.to(x.device)  # for some reason this is very slow

        x = torch.cat((x, last_outputs), dim=1)

        tmp_out = self.DG_projection(x)
        # log.info(tmp_out)
        if self.depth_sensor:
            depth_out = self.depth_encoder(obs_dict["obs"][:, -1:, :, :])
            depth_out = depth_out.view(obs_dict["obs"].size(0), -1)
            bypass_out = torch.cat((depth_out, last_outputs), dim=1)
        else:
            bypass_out = x

        if self.bypass:
            tmp_out = torch.cat((tmp_out, bypass_out), dim=1)
        elif hasattr(self, "dense"):
            dense_out = self.dense(x)
            tmp_out = torch.cat((tmp_out, dense_out), dim=1)

        return tmp_out

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ResNet18Layer2(Encoder):
    """
    Use ResNet-18 up to layer2 (i.e., conv1/bn1/relu/maxpool -> layer1 -> layer2),
    then a downsample (AvgPool2d) and flatten.
    """
    def __init__(self, cfg, obs_space, pretrained=True, fixed=True, down_k=3, down_s=2, down_p=1):
        super().__init__(cfg)

        in_ch = obs_space.shape[0]
        if in_ch != 3:
            raise NotImplementedError("FixedResNet18EarlyEncoder only supports 3-channel inputs.")

        # --- Weights handling across torchvision versions
        self._use_legacy_pretrained = False
        if pretrained:
            if hasattr(models, "ResNet18_Weights"):
                WeightsEnum = models.ResNet18_Weights
                weights_arg = WeightsEnum.IMAGENET1K_V1
            else:
                weights_arg = None
                self._use_legacy_pretrained = True
        else:
            weights_arg = None

        # --- Load ResNet-18
        if self._use_legacy_pretrained:
            resnet = models.resnet18(pretrained=pretrained)
        else:
            resnet = models.resnet18(weights=weights_arg)

        # --- Freeze if requested
        if fixed:
            for p in resnet.parameters():
                p.requires_grad = False
            resnet.eval()

        # --- Build the feature trunk up to layer2
        stem = nn.Sequential(
            resnet.conv1,   # 64, stride=2, kernel=7
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # /4 spatial after stem (given stride/pool)
        )
        self.features = nn.Sequential(
            stem,           # -> [N, 64, H/4, W/4] roughly
            resnet.layer1,  # -> [N, 64, ...]
            resnet.layer2,  # -> [N, 128, ...]
        )

        # --- Lightweight downsample (adjustable)
        self.downsample = nn.AvgPool2d(kernel_size=down_k, stride=down_s, padding=down_p)

        # --- ImageNet per-channel mean/std (for Normalize-like behavior)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        self.register_buffer("mn_mean", mean)
        self.register_buffer("mn_std",  std)

        # --- Compute encoder_out_size dynamically from obs_space
        _, H, W = obs_space.shape
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            y = (dummy - self.mn_mean) / self.mn_std
            y = self.features(y)
            y = self.downsample(y)
            self.encoder_out_size = y.numel()  # batch=1

        # --- Move to device
        device = "cuda" if getattr(cfg, "device", "").lower() in ("cuda", "gpu") else "cpu"
        self.model_to_device(device)

    def forward(self, x):
        # If upstream obs are raw [0,1], this matches ImageNet normalization.
        # If upstream obs are already zero-mean/unit-std, consider disabling that upstream norm.
        x = (x - self.mn_mean) / self.mn_std
        x = self.features(x)      # [N, 128, H', W']
        x = self.downsample(x)    # [N, 128, H'', W'']
        return torch.flatten(x, 1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


import cv2
import numpy as np
import torch.nn.functional as F

class RandomGaborEncoder(Encoder):
    """
    Visual encoder that begins with a frozen bank of Gabor filters (randomized orientations
    and scales), followed by a small stack of learnable convolutions to detect geometric
    compositions (e.g. walls, corners) in DeepMind Lab scenes.
    """
    def __init__(self, cfg, obs_space, pretrained=False, fixed=True):
        super().__init__(cfg)
        in_ch = obs_space.shape[0]
        # -- Gabor bank parameters (can be overridden via cfg) --
        num_scales = getattr(cfg, 'gabor_scales', 3)
        num_orient = getattr(cfg, 'gabor_orientations', 8)
        ksize = getattr(cfg, 'gabor_kernel_size', 21)
        # build frequency & orientation lists
        freqs = np.linspace(0.05, 0.4, num_scales)
        thetas = np.linspace(0, np.pi, num_orient, endpoint=False)
        # total filter count
        self.num_gabor = num_scales * num_orient
        # create Gabor kernels
        kernels = []
        for freq in freqs:
            for theta in thetas:
                kern = cv2.getGaborKernel(
                    ksize=(ksize, ksize),
                    sigma=ksize/6.0,
                    theta=theta,
                    lambd=ksize/(2.0*freq),
                    gamma=0.5,
                    psi=0
                ).astype(np.float32)
                kernels.append(kern)
        bank = np.stack(kernels, axis=0)                   # [K, H, W]
        bank = bank[:, None, :, :]                         # [K,1,H,W]
        bank = np.repeat(bank, in_ch, axis=1)              # [K,in_ch,H,W]
        # define frozen conv with Gabor weights
        self.gabor_conv = nn.Conv2d(
            in_ch, self.num_gabor,
            kernel_size=ksize, padding=ksize//2, bias=False
        )
        with torch.no_grad():
            self.gabor_conv.weight.copy_(torch.from_numpy(bank))
        for p in self.gabor_conv.parameters(): p.requires_grad = False

        # -- Learnable convolutional stack to compose shape features --
        mid_ch = getattr(cfg, 'gabor_mid_channels', 16)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(self.num_gabor, mid_ch, stride=2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, stride=2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # optionally freeze or fine-tune
        if fixed:
            for p in self.conv_stack.parameters(): p.requires_grad = False
        # global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # output feature dimension
        # self.encoder_out_size = mid_ch*96*72/2/2
        # self.measure = nn.Sequential(
        #     self.gabor_conv,
        #     self.conv_stack
        # )
        # self.conv_head_out_size = calc_num_elements(self.measure, obs_space.shape)
        self.encoder_out_size   = mid_ch * (96//8)*(72//8)  # assuming input is 96x72, adjust if different

        # move to device
        device = 'cuda' if getattr(cfg, 'device','').lower() in ('cuda','gpu') else 'cpu'
        self.model_to_device(device)

    def forward(self, x):
        # x: [N, C, H, W], float in [0..1] or pre-normalized
        # apply frozen Gabor bank
        x = self.gabor_conv(x)
        x = F.relu(x)
        # learnable composition convs
        x = self.conv_stack(x)
        # global pooling
        x = self.downsample(x)
        return x.flatten(1)

    def get_out_size(self):
        return self.encoder_out_size




def make_img_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Make (most likely convolutional) encoder for image-based observations."""
    if cfg.encoder_conv_architecture.startswith("convnet"):
        return ConvEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("resnet"):
        return ResnetEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("pretrained_resnet"):
        # Load the checkpoint.
        if cfg.encoder_load_path:
            encoder_load_path = cfg.encoder_load_path
        else:
            # this loads the SS RNN trained encoder
            encoder_load_path = "/home/xiaoxiong/try0120/train_dir/Random3_resnet_DG_relu_SS_RNN/checkpoint_p2/best_000020923_170811392_reward_87.534.pth"
        devicename = cfg.device
        if devicename == "gpu":
            devicename = "cuda"
        checkpoint = torch.load(encoder_load_path, map_location=devicename, weights_only=False)

        full_state_dict = checkpoint["model"]

        # Filter out only the keys for the encoder.
        encoder_state_dict = {
            k.replace("encoder.basic_encoder.", ""): v
            for k, v in full_state_dict.items()
            if k.startswith("encoder.basic_encoder.")
        }

        # Now create a new encoder instance. Note that pretrained is set to False because you'll load your custom weights,
        # and fixed is True to freeze the encoder.
        encoder = ResnetEncoder(cfg, obs_space)

        # Load the encoder state dict into the new encoder instance.
        encoder.load_state_dict(encoder_state_dict)

        if cfg.fix_encoder_when_load:
            log.info("fix encoder weights")
            # Double-check that the encoder parameters are frozen.
            for param in encoder.parameters():
                param.requires_grad = False

            encoder.eval()  # Make sure the encoder is in evaluation mode.
        else:
            log.info("trainable loaded encoder")

        return encoder
    elif cfg.encoder_conv_architecture.startswith("mobilenet"):
        return FixedMobileNetSmallEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("layer2_resnet18"):
        return ResNet18Layer2(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("gabor"):
        return RandomGaborEncoder(cfg, obs_space)
    else:
        raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_architecture}")


def make_hipposlam_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    if cfg.encoder_name == "Default":
        return DmlabEncoder(cfg, obs_space)
    return HipposlamEncoder(cfg, obs_space)
