local custom_observations = require 'decorators.custom_observations'
local debug_observations = require 'decorators.debug_observations'
local make_map = require 'common.make_map'
local maze_generation = require 'dmlab.system.maze_generation'
local pickups = require 'common.pickups'
local pickups_spawn = require 'dmlab.system.pickups_spawn'
local game = require 'dmlab.system.game'
local random = require 'common.random'
local themes = require 'themes.themes'
local decals = require 'themes.decals'
local setting_overrides = require 'decorators.setting_overrides'
local tensor = require 'dmlab.system.tensor'

local MAP_BASE = [[
*******
*     *
*  P  *
*  *  *
*  *  *
*  *  *
*  *  *
*  *  *
*  *  *
*******
]]

local api = {}

-- WITH INSTRUCTIONS AND outcome_event PULSE, prev_trial_reward PERSISTENT, chosen_arm PERSISTENT, and frameskip countdown for outcome_event pulse at trigger site. 

-------------
--- ITEMS ---
-------------

local Transparents = {
  name = 'Transparent',
  classname = 'transparent',
  model = 'models/goal_transparent.md3',
  quantity = 0,
  type = pickups.type.REWARD,
  wait = 0.5
}

local Big = {
  name = 'Big',
  classname = 'rewardR',
  model = 'models/goal_transparent.md3',
  quantity = 0,
  type = pickups.type.REWARD,
  wait = 0.5
}

local Small = {
  name = 'Small',
  classname = 'rewardL',
  model = 'models/goal_transparent.md3',
  quantity = 0,
  type = pickups.type.REWARD,
  wait = 0.5
}

function api:registerDynamicItems()
  return {'rewardR', 'rewardL', 'transparent'}
end

function api:extraEntities()
  return {
    {
      classname = 'transparent',
      origin = '350 859 17',
      count = '0',
      id = '1',
    },
  }
end

---------------------------------------------
--- RUN EPISODE, ITEM PICKUP AND SPAWNING ---
---------------------------------------------

local RIGHT_REWARD_ZONE = {
    xmin = 460, xmax = 540,
    ymin = 210, ymax = 300,
}

local LEFT_REWARD_ZONE = {
    xmin = 160, xmax = 240,
    ymin = 210, ymax = 300,
}

local ORIGIN_RIGHT   = '500 150 30'
local ORIGIN_LEFT    = '200 150 30'
local ORIGIN_TRIGGER = '350 859 17'

local FRAMESKIP = 4

local function inZone(pos, zone)
    return pos[1] >= zone.xmin and pos[1] <= zone.xmax
       and pos[2] >= zone.ymin and pos[2] <= zone.ymax
end

local function sampleOutcome(probability)
    local roll = random:uniformReal(0.0, 1.0)

    if roll <= probability then
        return {
            value = 10,
            outcome = "hit",
            roll = roll,
        }
    end

    return {
        value = 0,
        outcome = "miss",
        roll = roll,
    }
end

function api:start(episode, seed, params)
  math.randomseed(seed)
  api._trial_active = false
  api._trial_count = 1
  api._block_count = 1
  api._prob_block = "right" -- Right block status, "base" (80% Right, 20% Left)
  api._inst_block = 1

  api._reward_input_pulse = 0.0
  api._persistent_reward = 0.0
  api._outcome_event_pulse = 0.0
  api._outcome_timer = 0

  api._chosen_arm = 0.0 -- Persistent until next choice: 1.0 = Right, 2.0 = Left, 0.0 = None
  api._trial_committed = false

  api._highrew_hit_count = 0.0
  api._highrew_miss_count = 0.0
  api._lowrew_hit_count = 0.0
  api._lowrew_miss_count = 0.0

  cur_highrew_hit = 0.0
  cur_highrew_miss = 0.0
  cur_lowrew_hit = 0.0
  cur_lowrew_miss = 0.0

  api._base_right_count = 0.0
  api._base_left_count = 0.0
  api._rev_right_count = 0.0
  api._rev_left_count = 0.0
  api._flexibility = 0.0

  api._approach_pulsed = false
  api._in_reward_zone = false

  api._winning_arm = nil
  api._planned_outcome_value = 0

  local current_time = game:episodeTimeSeconds()
  api._episode_start_deadline = current_time + (5 * 60)
  api._iti_deadline = nil
  
  print("\n==========================================")
  print("--- BLOCK " .. api._block_count .. " STARTED (High: RIGHT 80%, Low: LEFT 20%) ---")
  print("==========================================")
end

function api:modifyControl(controls)
  -- Outcome event pulse timer countdown (held for 1 frameskip cycle at trigger site)
  if api._outcome_timer and api._outcome_timer > 0 then
      api._outcome_timer = api._outcome_timer - 1
      if api._outcome_timer == 0 then
          api._outcome_event_pulse = 0.0
      end
  end

  local current_time = game:episodeTimeSeconds()

  -- Trial is NOT active (waiting for agent to find trigger)
  if not api._trial_active then
      if api._episode_start_deadline and current_time > api._episode_start_deadline then
          game:console('setviewpos 345 714 25 90')
          api._episode_start_deadline = current_time + (5 * 60)
      elseif api._iti_deadline and current_time > api._iti_deadline then
          game:console('setviewpos 345 714 25 90')
          api._iti_deadline = current_time + (5 * 60)
      end

      return controls
  end

  -- If arm already committed for this trial, do not allow switching
  if api._trial_committed then
      return controls
  end

  local pos = game:playerInfo().pos

  local entered_right_zone = inZone(pos, RIGHT_REWARD_ZONE) and not api._in_reward_zone
  local entered_left_zone  = inZone(pos, LEFT_REWARD_ZONE) and not api._in_reward_zone

  if entered_right_zone then
    print("Entered RIGHT reward zone!")
    api._trial_committed = true
    api._chosen_arm = 1.0 -- Persistent for chosen arm (Right)
    api._approach_pulsed = true
    api._in_reward_zone = true

    if api._winning_arm == "right" then
      print("Reward predicted RIGHT (+10)")
      api._reward_input_pulse = 1.0
      api._planned_outcome_value = 10
    else
      print("Miss on RIGHT (0)")
      api._reward_input_pulse = 0.0
      api._planned_outcome_value = 0
    end

    -- Spawn endpoint pickup ONLY for Right arm
    pickups_spawn:spawn{
        classname = 'rewardR',
        origin = ORIGIN_RIGHT,
        id = '2'
    }

  elseif entered_left_zone then
    print("Entered LEFT reward zone!")
    api._trial_committed = true
    api._chosen_arm = 2.0 -- Persistent for chosen arm (Left)
    api._approach_pulsed = true
    api._in_reward_zone = true

    if api._winning_arm == "left" then
      print("Reward predicted LEFT (+10)")
      api._reward_input_pulse = 1.0
      api._planned_outcome_value = 10
    else
      print("Miss on LEFT (0)")
      api._reward_input_pulse = 0.0
      api._planned_outcome_value = 0
    end

    -- Spawn endpoint pickup ONLY for Left arm
    pickups_spawn:spawn{
        classname = 'rewardL',
        origin = ORIGIN_LEFT,
        id = '3'
    }
  else
    api._in_reward_zone = false
  end

  return controls
end

function api:hasEpisodeFinished(timeSeconds)
    if api._block_count > 5 then
        print("==========================================")
        print("--- 5 BLOCKS COMPLETED ---")
        print("==========================================")
        return true
    end
    return false
end

function api:createPickup(classname) 
  if classname == 'transparent' then
    return Transparents
  elseif classname == 'rewardR' then
    return Big
  elseif classname == 'rewardL' then
    return Small
  end
  
  return default_pickup
end

function api:pickup(id, playerId)
  local current_time = game:episodeTimeSeconds()

  -- TRIGGER PICKED UP (Start of Trial / Outcome of Previous Trial Processed Here)
  if id == 1 then
    api._approach_pulsed = false
    api._in_reward_zone = false

    -- Reset approach pulse after trigger is picked up
    api._reward_input_pulse = 0.0

    if not api._trial_active then
        api._trial_active = true
        api._trial_committed = false       -- Reset commitment lock for the new trial

        -- Trigger outcome_event pulse at the trigger site, latched for 1 frameskip cycle
        api._outcome_event_pulse = 1.0
        api._outcome_timer = FRAMESKIP
        api._trigger_time = current_time

        api._episode_start_deadline = nil
        api._iti_deadline = nil

        api.setInstruction(tostring(api._inst_block))

        -- SINGLE GLOBAL 80/20 LOTTERY VIA sampleOutcome
        local result = sampleOutcome(0.80)
        if result.outcome == "hit" then
            api._winning_arm = api._prob_block
        else
            api._winning_arm = (api._prob_block == "right") and "left" or "right"
        end

        print(string.format("[Trial %d | Block %d (%s)] Winning Arm: %s (Roll: %.2f)",
            api._trial_count, api._block_count, api._prob_block, api._winning_arm, result.roll))

        -- Clean up any lingering endpoint pickups
        game:console('dm_pickup 2')
        game:console('dm_pickup 3')
    end
    
  -- ENDPOINT PICKUP REACHED (Choice Made, Reward Delivered)
  elseif id == 2 or id == 3 then
    if api._trial_active then
        api._trial_active = false

        api._reward_time = current_time
        api._iti_deadline = current_time + (5 * 60)

        cur_highrew_hit = 0.0
        cur_highrew_miss = 0.0
        cur_lowrew_hit = 0.0
        cur_lowrew_miss = 0.0

        local is_hit = (api._winning_arm == (id == 2 and "right" or "left"))
        local score = is_hit and 10 or 0

        game:addScore(score)

        -- Persistent reward: keeps track of the outcome for HL-RNN throughout next trial
        if score > 0 then
          api._persistent_reward = 1.0
        else
          api._persistent_reward = 0.0
        end

        -- Record High-Reward vs Low-Reward Hit / Miss
        local chose_high_arm = ((id == 2 and "right" or "left") == api._prob_block)

        if chose_high_arm then
          if is_hit then
            cur_highrew_hit = 1.0
            api._highrew_hit_count = api._highrew_hit_count + 1.0
          else
            cur_highrew_miss = 1.0
            api._highrew_miss_count = api._highrew_miss_count + 1.0
          end
        else
          if is_hit then
            cur_lowrew_hit = 1.0
            api._lowrew_hit_count = api._lowrew_hit_count + 1.0
          else
            cur_lowrew_miss = 1.0
            api._lowrew_miss_count = api._lowrew_miss_count + 1.0
          end
        end

        -- Record Baseline Block vs Reversal Block Choices for Flexibility
        if api._prob_block == "right" then
          -- BASELINE BLOCK (Right is high reward)
          if id == 2 then
            api._base_right_count = api._base_right_count + 1.0
          elseif id == 3 then
            api._base_left_count = api._base_left_count + 1.0
          end
        else
          -- REVERSAL BLOCK (Left is high reward)
          if id == 2 then
            api._rev_right_count = api._rev_right_count + 1.0
          elseif id == 3 then
            api._rev_left_count = api._rev_left_count + 1.0
          end
        end

        -- ##################################################
        -- Calculate Flexibility: P(Right | Base) vs P(Right | Rev)
        -- ##################################################
        local total_base = api._base_right_count + api._base_left_count
        local p_base = 0.0
        if total_base > 0 then
            p_base = api._base_right_count / total_base
        end

        local total_rev = api._rev_right_count + api._rev_left_count
        local p_rev = 0.0
        if total_rev > 0 then
            p_rev = api._rev_right_count / total_rev
        end

        -- Flexibility index = (P_base - P_rev) / (P_base + P_rev)
        -- Evaluated once both baseline and reversal blocks have occurred
        if total_base > 0 and total_rev > 0 and (p_base + p_rev) > 0 then
            api._flexibility = (p_base - p_rev) / (p_base + p_rev)
        else
            api._flexibility = 0.0
        end

        -- Despawn any active endpoint pickups
        game:console('dm_pickup 2')
        game:console('dm_pickup 3')

        -- BLOCK TRANSITION LOGIC --
        api._trial_count = api._trial_count + 1
        
        if (api._trial_count - 1) % 15 == 0 then
            print("==========================================")
            print("--- END OF BLOCK " .. api._block_count .. " (15 Trials Completed) ---")
            api._block_count = api._block_count + 1
            api._trial_count = 1 -- Reset trial count for the new block
            
            local shift_roll = random:uniformReal(0.0, 1.0)
            if shift_roll <= 0.66 then
                -- Flip the state
                if api._prob_block == "right" then
                     api._prob_block = "left"
                     api._inst_block = 2
                else
                     api._prob_block = "right"
                     api._inst_block = 1
                end
                print("[Shift Roll: " .. string.format("%.2f", shift_roll) .. " <= 0.66 -> [CONTINGENCY INVERSION FOR BLOCK " .. api._block_count .. "]]")
            else
                print("[Shift Roll: " .. string.format("%.2f", shift_roll) .. " > 0.66 -> [CONTINGENCIES REMAIN THE SAME FOR BLOCK " .. api._block_count .. "]]")
            end
            print("==========================================")
        end

        -- Respawn trigger for next trial
        pickups_spawn:spawn{
            classname = 'transparent',
            origin = ORIGIN_TRIGGER,
            count = '0',
            id = '1'
        }
    end
  end
end

function api:customObservationSpec()
    return {
        {name = 'highrew_hit', type = 'Doubles', shape = {1}},
        {name = 'highrew_miss', type = 'Doubles', shape = {1}},
        {name = 'lowrew_hit', type = 'Doubles', shape = {1}},
        {name = 'lowrew_miss', type = 'Doubles', shape = {1}},
        {name = 'highrew_hit_total', type = 'Doubles', shape = {1}},
        {name = 'highrew_miss_total', type = 'Doubles', shape = {1}},
        {name = 'lowrew_hit_total', type = 'Doubles', shape = {1}},
        {name = 'lowrew_miss_total', type = 'Doubles', shape = {1}},
        {name = 'flexibility', type = 'Doubles', shape = {1}},
        {name = 'reward_input', type = 'Doubles', shape = {1}},
        {name = 'inst_block', type = 'Doubles', shape = {1}},
        {name = 'outcome_event', type = 'Doubles', shape = {1}},
        {name = 'prev_trial_reward', type = 'Doubles', shape = {1}},
        {name = 'chosen_arm', type = 'Doubles', shape = {1}},
    }
end

function api:customObservation(name)
    if name == 'highrew_hit' then
        local val = cur_highrew_hit
        cur_highrew_hit = 0.0 -- Reset pulse immediately
        return tensor.Tensor{val}
    elseif name == 'highrew_miss' then
        local val = cur_highrew_miss
        cur_highrew_miss = 0.0
        return tensor.Tensor{val}
    elseif name == 'lowrew_hit' then
        local val = cur_lowrew_hit
        cur_lowrew_hit = 0.0
        return tensor.Tensor{val}
    elseif name == 'lowrew_miss' then
        local val = cur_lowrew_miss
        cur_lowrew_miss = 0.0
        return tensor.Tensor{val}
    -- Expose running totals for Python to grab at the end of the episode
    elseif name == 'highrew_hit_total' then
        return tensor.Tensor{api._highrew_hit_count}
    elseif name == 'highrew_miss_total' then
        return tensor.Tensor{api._highrew_miss_count}
    elseif name == 'lowrew_hit_total' then
        return tensor.Tensor{api._lowrew_hit_count}
    elseif name == 'lowrew_miss_total' then
        return tensor.Tensor{api._lowrew_miss_count}
    end
    if name == 'outcome_event' then
        local val = api._outcome_event_pulse
        return tensor.Tensor{val}
    end
    if name == 'prev_trial_reward' then
        return tensor.Tensor{api._persistent_reward}
    end
    if name == 'flexibility' then
        return tensor.Tensor{api._flexibility}
    end
    if name == 'reward_input' then
        local val = api._reward_input_pulse
        return tensor.Tensor{val}
    end
    if name == 'inst_block' then
        return tensor.Tensor{api._inst_block}
    end
    if name == 'chosen_arm' then
        local val = api._chosen_arm
        return tensor.Tensor{val}
    end
end

----------------------
--- MAP DECORATION ---
----------------------

local my_textures = {
    floor = {{tex = 'map/lab_games/cretebase'}},
    ceiling = {{tex = 'map/lab_games/cretebase'}},
    wall = {{tex = 'map/lab_games/cretebase'}},
    wallDecals = decals.decals,
}

function api:init(params)
  print("Initializing map...")

  local my_theme = themes.fromTextureSet{
        textureSet = my_textures,
        decalFrequency = 1, 
        floorModelFrequency = 1,
    }

    function my_theme:placeWallDecals(allWallLocations)
        local wallDecals = {}
        local decal_count = 1
        local available_decals = my_textures.wallDecals

        local total_walls = #allWallLocations
        
        for i = 1, total_walls do
            local current_index = allWallLocations[i].index
          
            if current_index == 9 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[7]
                }
                decal_count = decal_count + 1
                
            elseif current_index == 34 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[18] 
                }
                decal_count = decal_count + 1
            
            elseif current_index == 38 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[17] 
                }
                decal_count = decal_count + 1

            elseif current_index == 4 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[3] 
                }
                decal_count = decal_count + 1

            end
        end
      
        return wallDecals
    end

  api._map = make_map.makeMap{
      mapName = 'y_maze',
      mapEntityLayer = MAP_BASE,
      useSkybox = true,
      theme = my_theme,
  }
  print("Map created:", api._map)
end

function api:nextMap()
  local maze = maze_generation:mazeGeneration{entity = MAP_BASE}
  debug_observations.setMaze(maze)
  api.setInstruction(tostring(api._inst_block))
  return api._map 
end

setting_overrides.decorate{
    api = api,
    apiParams = {episodeLengthSeconds = 60 * 60, camera = {1050, 1050, 1000}},
    decorateWithTimeout = true
}
custom_observations.decorate(api)

return api