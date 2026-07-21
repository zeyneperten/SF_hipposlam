local custom_observations = require 'decorators.custom_observations'
local debug_observations = require 'decorators.debug_observations'
local make_map = require 'common.make_map'
local maze_generation = require 'dmlab.system.maze_generation'
local pickups = require 'common.pickups'
local pickups_spawn = require 'dmlab.system.pickups_spawn'
local game = require 'dmlab.system.game'
local game_entities = require 'dmlab.system.game_entities'
local random = require 'common.random'
local themes = require 'themes.themes'
local texture_sets = require 'themes.texture_sets'
local decals = require 'themes.decals'
local setting_overrides = require 'decorators.setting_overrides'

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
  classname = 'reward1',
  model = 'models/goal_transparent.md3',
  quantity = 0,
  type = pickups.type.REWARD,
  wait = 0.5
}

local Small = {
  name = 'Small',
  classname = 'reward2',
  model = 'models/goal_transparent.md3',
  quantity = 0,
  type = pickups.type.REWARD,
  wait = 0.5
}

function api:registerDynamicItems()
  return {'reward1', 'reward2', 'transparent'}
end

function api:extraEntities()
  return {
    {
      classname = 'transparent',
      origin = '350 859 17',
      count = '0',
      id = '1',
    },
    {
      classname = 'light_flame_large_yellow',
      origin = '200 150 30',
      light = '300',
      style = '0',
    }
  }
end

---------------------------------------------
--- RUN EPISODE, ITEM PICKUP AND SPAWNING ---
---------------------------------------------

function api:start(episode, seed, params)
  math.randomseed(seed)

  api._trial_active = false
  api._trial_count = 1
  api._block_count = 1
  api._prob_block = "baseline" -- Baseline block status
  --api._timer = 0

  local current_time = game:episodeTimeSeconds()
  
  api._episode_start_deadline = current_time + 10
  api._iti_deadline = nil
  
  print("\n==========================================")
  print("--- BLOCK " .. api._block_count .. " STARTED ---")
  print("==========================================")
end

function api:modifyControl(controls)
  local current_time = game:episodeTimeSeconds()

  -- Trial is NOT active (waiting for agent to find trigger)
  if not api._trial_active then
      
      -- Find trigger in the beginning
      if api._episode_start_deadline and current_time > api._episode_start_deadline then
          print("[Timeout] Initial startup omission: Failed to find first trigger. Resetting.")

          game:console('setviewpos 345 714 25 90')
          api._episode_start_deadline = current_time + 10 -- Refresh initial window if needed
      
      -- Find trigger after reward pickup
      elseif api._iti_deadline and current_time > api._iti_deadline then
          print("[Timeout] ITI Omission: Took too long to return to trigger after reward. Resetting.")

          game:console('setviewpos 345 714 25 90')
          api._iti_deadline = current_time + 10
      end
      
  end

  return controls
end


function api:createPickup(classname) 
  if classname == 'transparent' then
    return Transparents
  elseif classname == 'reward1' then
    return Big
  elseif classname == 'reward2' then
    return Small
  end
  
  return default_pickup
end

function api:pickup(id, playerId)
  local current_time = game:episodeTimeSeconds()

  if id == 1 then
    if not api._trial_active then
        api._trial_active = true
        api._trigger_time = current_time
        
        -- PERMANENTLY clear the initial startup deadline since the first trial has begun!
        api._episode_start_deadline = nil
        api._iti_deadline = nil

        if api._reward_time then
            local rw2tr_time = api._trigger_time - api._reward_time
            print(string.format("[Block %d | Trial %d] Retriggered! ITI duration (Reward to Trigger): %.2fs", api._block_count, api._trial_count, rw2tr_time))
        else
            print("[Block " .. api._block_count .. " | Trial " .. api._trial_count .. " Checkpoint] Trigger found! Took: " .. string.format("%.2f", api._trigger_time) .. "s")
        end

        -- 1. FIX THE ITEM LOCATIONS
        local origin_left  = '200 150 30'
        local origin_right = '500 150 30'
        
        pickups_spawn:spawn{
            classname = 'reward1',
            origin = origin_left,
            id = '2'
        }
        pickups_spawn:spawn{
            classname = 'reward2',
            origin = origin_right,
            id = '3'
        }
        
    end
    
  elseif id == 2 or id == 3 then
    if api._trial_active then
        api._trial_active = false

        api._reward_time = current_time -- total duration to reward pickup
        api._tr2rw_time = api._reward_time - api._trigger_time

        -- ITI: Give the agent a separate window to walk back to the trigger
        api._iti_deadline = current_time + 10
        
        -- PROBABILISTIC REWARD --
        local payout_roll = random:uniformReal(0.0, 1.0)
        
        if id == 2 then --  RIGHT
            print("[Block " .. api._block_count .. " | Trial " .. api._trial_count .. " Finalpoint] Navigated RIGHT. Took " .. string.format("%.2f", api._tr2rw_time) .."s from trigger.")
            
            if api._prob_block == "baseline" then
                -- Baseline: Right has 80% chance of big reward
                if payout_roll <= 0.80 then
                    print("Hit! +10")
                    game:addScore(10)
                else
                    print("Miss! (Unlucky 20%) 0")
                    game:addScore(0)
                end
            else
                -- Reversed: Right has 20% chance of big reward
                if payout_roll <= 0.20 then
                    print("Hit! (Lucky 20%) +10")
                    game:addScore(10)
                else
                    print("Miss! 0")
                    game:addScore(0)
                end
            end
            
        elseif id == 3 then -- LEFT
            print("[Block " .. api._block_count .. " | Trial " .. api._trial_count .. " Finalpoint] Navigated LEFT. Took " .. string.format("%.2f", api._tr2rw_time) .."s from trigger.")
            
            if api._prob_block == "baseline" then
                if payout_roll <= 0.20 then
                    print("Hit! (Lucky 20%) +10")
                    game:addScore(10)
                else
                    print("Miss! 0")
                    game:addScore(0)
                end
            else
                if payout_roll <= 0.80 then
                    print("Hit! +10")
                    game:addScore(10)
                else
                    print("Miss! (Unlucky 20%) 0")
                    game:addScore(0)
                end
            end
        end

        -- BLOCK TRANSITION LOGIC --
        api._trial_count = api._trial_count + 1
        
        if (api._trial_count - 1) % 5 == 0 then
            print("==========================================")
            print("--- END OF BLOCK " .. api._block_count .. " (5 Trials Completed) ---")
            api._block_count = api._block_count + 1
            api._trial_count = 1 -- Reset trial count for the new block
            
            local shift_roll = random:uniformReal(0.0, 1.0)
            if shift_roll <= 0.66 then
                -- Flip the state
                if api._prob_block == "baseline" then
                     api._prob_block = "reversed"
                else
                     api._prob_block = "baseline"
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
            origin = '350 859 17',
            count = '0',
            id = '1'
        }
    end
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
  -- make_map.seedRng(4)  -- no need because we have fixed map and no randomization

  local my_theme = themes.fromTextureSet{
        textureSet = my_textures,
        decalFrequency = 1, 
        floorModelFrequency = 1,
    }

    function my_theme:placeWallDecals(allWallLocations)
        local wallDecals = {}
        local decal_count = 1 -- Counter to build the returned array perfectly
        local available_decals = my_textures.wallDecals

        --print("--- START WALL SCANNER ---")
        local total_walls = #allWallLocations
        --print("Total walls detected in engine: " .. total_walls)
        
        for i = 1, total_walls do
            local current_index = allWallLocations[i].index
          
            if current_index == 9 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[7] -- Assign a specific decal
                }
                decal_count = decal_count + 1
                
            elseif current_index == 34 or current_index == 38 then
                wallDecals[decal_count] = {
                    index = current_index,
                    decal = available_decals[18] 
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

        --print("--- END WALL SCANNER ---")
      
        return wallDecals
    end

  api._map = make_map.makeMap{
      mapName = 'Y maze',
      mapEntityLayer = MAP_BASE,
      useSkybox = true,
      --skyBoxTexture = my_textures.ceiling[1].tex,
      theme = my_theme,
  }
  print("Map created:", api._map)
end

function api:nextMap()
  local maze = maze_generation:mazeGeneration{entity = MAP_BASE}
  debug_observations.setMaze(maze)
  return api._map 
end
setting_overrides.decorate{
    api = api,
    apiParams = {episodeLengthSeconds = 60 * 60, camera = {1050, 1050, 1000}},
    decorateWithTimeout = true
}
custom_observations.decorate(api)

return api