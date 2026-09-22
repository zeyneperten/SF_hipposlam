from typing import Dict
from sample_factory.model.actor_critic import default_make_actor_critic_func, ActorCritic

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

def make_hipposlam_actor_critic(cfg, obs_space, action_space) -> ActorCritic:
    # Use Sample Factory's default creation logic
    actor_critic = default_make_actor_critic_func(cfg, obs_space, action_space)
    
    # Wrap it to expose our custom stats
    return add_custom_summaries(actor_critic)