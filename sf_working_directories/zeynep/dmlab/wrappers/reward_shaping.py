from math import tanh

import gymnasium as gym

from sample_factory.utils.utils import log

RAW_SCORE_SUMMARY_KEY_SUFFIX = "dmlab_raw_score"


class DmlabRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0

        ## ADDED ##
        self.hi_hit_count = self.hi_miss_count = self.lo_hit_count = self.lo_miss_count = 0
        self.hi_hit_history, self.hi_miss_history = [], []
        self.lo_hit_history, self.lo_miss_history = [], []
        ##########

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_episode_return = self.episode_length = 0

        ## ADDED ##
        # Clear trackers at the start of a new episode
        self.hi_hit_count = self.hi_miss_count = self.lo_hit_count = self.lo_miss_count = 0
        self.hi_hit_history, self.hi_miss_history = [], []
        self.lo_hit_history, self.lo_miss_history = [], []
        ###########
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        self.raw_episode_return += rew
        self.episode_length += info.get("num_frames", 1)

        #### ADDED: Continuous Tracking of Custom Metrics ####
        # --- CONTINUOUS STEP-BY-STEP TRACKING ---
        if info.get("highrew_hit"):
            self.hi_hit_count += 1
            self.hi_hit_history.append(self.episode_length)
            #log.warning(f"🚨 [FRAME {self.episode_length}] HIGH-REWARD HIT! (Total this episode: {self.hi_hit_count})")
            
        if info.get("highrew_miss"):
            self.hi_miss_count += 1
            self.hi_miss_history.append(self.episode_length)
            #log.warning(f"🚨 [FRAME {self.episode_length}] HIGH-REWARD MISS! (Total this episode: {self.hi_miss_count})")
            
        if info.get("lowrew_hit"):
            self.lo_hit_count += 1
            self.lo_hit_history.append(self.episode_length)
            #log.warning(f"🚨 [FRAME {self.episode_length}] LOW-REWARD HIT! (Total this episode: {self.lo_hit_count})")
            
        if info.get("lowrew_miss"):
            self.lo_miss_count += 1
            self.lo_miss_history.append(self.episode_length)
            #log.warning(f"🚨 [FRAME {self.episode_length}] LOW-REWARD MISS! (Total this episode: {self.lo_miss_count})")
        #####################################################
        

        # optimistic asymmetric clipping from IMPALA paper
        #squeezed = tanh(rew / 5.0)
        #clipped = 0.3 * squeezed if rew < 0.0 else squeezed
        #rew = clipped * 5.0

        if done:
            score = self.raw_episode_return

            # ADDED to Catch the stats coming up from dmlab_gym before they are erased!
            existing_extra_stats = info.get("episode_extra_stats", dict())
            #############

            info["episode_extra_stats"] = dict()
            level_name = self.unwrapped.level_name

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f"z_{self.unwrapped.task_id:02d}_{level_name}"
            info["episode_extra_stats"][f"{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}"] = score
            info["episode_extra_stats"][f"{level_name_key}_len"] = self.episode_length
            info["episode_extra_stats"][f"{level_name_key}_lenweighted_score"] = (
                (10000 - self.episode_length) / 10000 * score
            )

            ## ADDED ##
            #log.warning(
            #    "🏁 EPISODE FINISHED | Final Totals -> Hi-Hit: %s | Hi-Miss: %s | Lo-Hit: %s | Lo-Miss: %s",
            #    self.hi_hit_count, self.hi_miss_count, self.lo_hit_count, self.lo_miss_count
            #)

            info["episode_extra_stats"]["custom/highrew_hit"] = float(self.hi_hit_count)
            info["episode_extra_stats"]["custom/highrew_miss"] = float(self.hi_miss_count)
            info["episode_extra_stats"]["custom/lowrew_hit"] = float(self.lo_hit_count)
            info["episode_extra_stats"]["custom/lowrew_miss"] = float(self.lo_miss_count)

            info["episode_extra_stats"]["custom/flexibility"] = existing_extra_stats.get("custom/flexibility", 0.0)
            info["episode_extra_stats"]["custom/instr_switch"] = existing_extra_stats.get("custom/instr_switch", 1)
            
            # --- PACKAGE TIMELINES FOR YOUR LOCAL THESIS DATA ---
            info["hi_hit_history"] = self.hi_hit_history.copy()
            info["hi_miss_history"] = self.hi_miss_history.copy()
            info["lo_hit_history"] = self.lo_hit_history.copy()
            info["lo_miss_history"] = self.lo_miss_history.copy()
            ##########

            # log.info(f'Episode Extra Stats: {info["episode_extra_stats"]}')
        return obs, rew, terminated, truncated, info
