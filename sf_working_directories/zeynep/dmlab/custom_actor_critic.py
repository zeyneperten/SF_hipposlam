from typing import Dict
from sample_factory.model.actor_critic import default_make_actor_critic_func, ActorCritic
import torch
from sample_factory.utils.utils import log

def add_custom_summaries(actor_critic: ActorCritic) -> ActorCritic:
    """Dynamically extends the summaries method to pull custom stats from the Core."""
    original_summaries = actor_critic.summaries
    
    def custom_summaries() -> Dict:
        s = original_summaries()
        
        # Locate the core
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))
            
        # Extract our variables!
        if core is not None and hasattr(core, 'last_v_L'):
            s['context_rnn/v_L'] = core.last_v_L
            s['context_rnn/v_R'] = core.last_v_R
            s['context_rnn/v_diff'] = core.last_v_diff
            
            # Log the learned decay parameter if it exists
            if hasattr(core, 'context_rnn'):
                if hasattr(core.context_rnn, 'alpha'):
                    s['context_rnn/alpha'] = float(core.context_rnn.alpha.mean().item())
                elif hasattr(core.context_rnn, 'current_alpha'):
                    s['context_rnn/current_alpha'] = float(core.context_rnn.current_alpha.mean().item())
        return s
        
    actor_critic.summaries = custom_summaries
    return actor_critic

def HighLevel_LossWrapper(actor_critic: ActorCritic) -> ActorCritic:
    """Dynamically extends the forward method to include Q-learning or policy loss from the high-level RNN."""
    original_forward = actor_critic.forward
    original_summaries = actor_critic.summaries
    
    def custom_forward(*args, **kwargs):
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))
        # Value bootstrap must follow the same mode policy as acting. A greedy
        # mode here biases the target when the actor samples at trial boundaries.
        result_dict = original_forward(*args, **kwargs)

        if getattr(actor_critic.cfg, 'core_name', None) == 'BypassSS_HighLevelRNN':
            # This is the mode actually used by the actor's decoder at this step.
            # Sample Factory records it alongside actions for learner replay.
            result_dict['hl_z'] = result_dict['new_rnn_states'][:, -actor_critic.cfg.hl_K:].clone()

            #log.debug(f"High-level RNN mode (hl_z) shape: {result_dict['hl_z'].shape}, values: {result_dict['hl_z']}")
        
        # 2. Locate the core
        # 3. Extract the Q-loss if it exists
        if core is not None and hasattr(core, 'last_hl_loss'):
            result_dict['hl_loss'] = core.last_hl_loss
        else:
            # Fallback for inference pass to prevent crashes
            # (Grabs the device from the 'values' tensor safely)
            device = result_dict['values'].device if 'values' in result_dict else torch.device('cpu')
            result_dict['hl_loss'] = torch.tensor(0.0, device=device)
        
        return result_dict
    
    def summaries_with_hl_loss() -> Dict:
        s = original_summaries()
        core = getattr(actor_critic, 'core', getattr(actor_critic, 'actor_core', None))

        if core is not None:
            # Add high-level loss to the summaries
            if hasattr(core, 'last_hl_loss') and core.last_hl_loss is not None:
                s['hl/hl_loss'] = core.last_hl_loss.item() # Use .item() for logging!

            if hasattr(core, 'last_hl_metrics') and core.last_hl_metrics is not None:
                for key, value in core.last_hl_metrics.items():
                    s[key] = value
                
            if hasattr(core, 'last_log_dict') and core.last_log_dict is not None:
                for key, value in core.last_log_dict.items():
                    if isinstance(value, torch.Tensor):
                        s[f'hl/{key}'] = value.float().mean().item()
                    else:
                        s[f'hl/{key}'] = value
        return s
    
    # Override the forward method
    actor_critic.forward = custom_forward
    actor_critic.summaries = summaries_with_hl_loss
    return actor_critic


def make_hipposlam_actor_critic(cfg, obs_space, action_space) -> ActorCritic:
    # Use Sample Factory's default creation logic
    actor_critic = default_make_actor_critic_func(cfg, obs_space, action_space)

    if getattr(cfg, "hl_train_fix_base", False):
        log.warning("Fix encoder, base core, decoder. Only train high-level RNN and its loss.")

        # 1. Freeze the vision
        if hasattr(actor_critic, 'encoder'):
            for param in actor_critic.encoder.parameters():
                param.requires_grad = False
        
        if hasattr(actor_critic, 'core'):
            if hasattr(actor_critic.core, 'base_core'):
                for param in actor_critic.core.base_core.parameters():
                    param.requires_grad = False
            else:
                log.warning("No base_core found in actor_critic.core. Skipping freezing base_core parameters.")

        if hasattr(actor_critic, 'decoder'):
            for param in actor_critic.decoder.parameters():
                param.requires_grad = False



    if getattr(cfg, 'hl_diversity_reward_coef', 0.0) > 0.0:
        if cfg.core_name != 'BypassSS_HighLevelRNN':
            raise ValueError('High-level diversity reward requires BypassSS_HighLevelRNN')
        from sf_working_directories.zeynep.dmlab.high_level_diversity import TrialEndModeClassifier

        actor_critic.hl_diversity_classifier = TrialEndModeClassifier(
            image_channels=obs_space['obs'].shape[0],
            num_modes=cfg.hl_K,
            include_chosen_arm=cfg.hl_diversity_include_chosen_arm,
        )
    return HighLevel_LossWrapper(actor_critic)
    #return add_custom_summaries(actor_critic)
