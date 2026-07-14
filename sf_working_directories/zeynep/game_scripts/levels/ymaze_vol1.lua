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

local my_custom_textures = {
    floor = {{tex = 'map/lab_games/cretebase'}},
    ceiling = {{tex = 'map/lab_games/sky/lg_sky_01_dn'}},
    wall = {{tex = 'map/lab_games/cretebase'}},
}

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

function api:init(params)
  print("Initializing map...")
  make_map.seedRng(4) 
  api._map = make_map.makeMap{
      mapName = 'Y maze',
      mapEntityLayer = MAP_BASE,
      useSkybox = true,
      theme = themes.fromTextureSet{
          textureSet = my_custom_textures,
          decalFrequency = 0.05,
          floorModelFrequency = 1,
      },
  }
  print("Map created: ", api._map)
end

function api:extraEntities()
  return {
    {
      classname = 'transparent',
      origin = '350 850 17',
      count = '0',
      id = '1',
    }
  }
end

function api:start(episode, seed, params)
  print("Episode started")
  api._trial_active = false
  math.randomseed(seed)
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
  if id == 1 then
    if not api._trial_active then
        print('Trigger picked! Trial Started.')
        api._trial_active = true

        local coord_1 = '200 150 30'
        local coord_2 = '500 150 30'
        
        local origin_reward1 = ''
        local origin_reward2 = ''
        
        rand_num=random:uniformInt(1, 2)
        
        if rand_num == 1 then
            origin_reward1 = coord_1
            origin_reward2 = coord_2
            print('Spawn Layout: Small Left, Big Right')
        else
            origin_reward1 = coord_2
            origin_reward2 = coord_1
            print('Spawn Layout: Big Left, Small Right')
        end

        pickups_spawn:spawn{
            classname = 'reward1',
            origin = origin_reward1,
            id = '2'
        }
        pickups_spawn:spawn{
            classname = 'reward2',
            origin = origin_reward2,
            id = '3'
        }
    end
    
  elseif id == 2 or id == 3 then
    if api._trial_active then
        print('Choice made! Resetting trial state.')
        api._trial_active = false
        
        if id == 2 then
            print("Correct! +10")
            game:addScore(10)
            game:console('dm_pickup 3')

        elseif id == 3 then
            print("Wrong! 0")
            game:addScore(0)
            game:console('dm_pickup 2')
        end

        print('Respawning the trigger for the next round...')
        pickups_spawn:spawn{
            classname = 'transparent',
            origin = '350 850 17',
            count = '0',
            id = '1'
        }
    end
  end
end

function api:nextMap()
  local maze = maze_generation:mazeGeneration{entity = MAP_BASE}
  debug_observations.setMaze(maze)
  return api._map 
end

custom_observations.decorate(api)

return api