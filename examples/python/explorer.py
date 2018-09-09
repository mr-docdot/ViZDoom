#!/usr/bin/env python3

#####################################################################
# This script presents how to use the environment with PyOblige.
# https://github.com/mwydmuch/PyOblige
#####################################################################

from __future__ import print_function
from time import sleep
import os

import itertools as it
import random
import vizdoom as vzd
from vizdoom import Button
from argparse import ArgumentParser
import oblige
import cv2
import numpy as np
import math

DEFAULT_CONFIG = "../../scenarios/explorer.cfg"
DEFAULT_SEED = 999
DEFAULT_OUTPUT_FILE = "gen_scene_%d.wad" % (DEFAULT_SEED)

def spin_agent(actions_num):
    actions = [ 0.0 ] * actions_num
    actions[8] = 1.0
    return actions

def compute_map(state, height=960, width=1280,
               map_size=256, map_scale=3, fov=90.0):

    depth_buffer = state.depth_buffer

    player_x = state.game_variables[5]
    player_y = state.game_variables[6]
    player_z = state.game_variables[7]

    player_angle = state.game_variables[8]
    player_pitch = state.game_variables[9]
    player_roll = state.game_variables[10]

    canvas_size = 2*map_size + 1
    vis_map = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    simple_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    r = canvas_size
    offset = 225

    y1 = int(r * math.cos(math.radians(offset + player_angle)))
    x1 = int(r * math.sin(math.radians(offset + player_angle)))

    y2 = int(r * math.cos(math.radians(offset + player_angle - fov)))
    x2 = int(r * math.sin(math.radians(offset + player_angle - fov)))

    _, p1, p2 = cv2.clipLine((0, 0, canvas_size, canvas_size), (map_size, map_size),
                             (map_size + x1, map_size + y1))
    _, p3, p4 = cv2.clipLine((0, 0, canvas_size, canvas_size), (map_size, map_size),
                             (map_size + x2, map_size + y2))

    game_unit = 110.0/14
    ray_cast = (depth_buffer[height/2] * game_unit)/float(map_scale)

    ray_points = [ (map_size, map_size) ]
    for i in range(canvas_size):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size), (map_size, map_size),
                                                          (ray_y, ray_x))
        ray_points.append(p)

    cv2.fillPoly(vis_map, np.array([ray_points], dtype=np.int32), (255, 255, 255))
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

    for l in state.labels:
        object_relative_x = -l.object_position_x + player_x
        object_relative_y = -l.object_position_y + player_y
        object_relative_z = -l.object_position_z + player_z

        print(l.object_name, (object_relative_x),
              (object_relative_y),
              (object_relative_z))

        scaled_x = object_relative_x
        scaled_y = object_relative_y

        rotated_x = math.cos(math.radians(-player_angle)) * scaled_x - math.sin(math.radians(-player_angle)) * scaled_y
        rotated_y = math.sin(math.radians(-player_angle)) * scaled_x + math.cos(math.radians(-player_angle)) * scaled_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            color = (0, 0, 255)
            object_id = 2
            simple_map[rotated_y, rotated_x] = object_id
            cv2.circle(vis_map, (rotated_y, rotated_x), 2, color, thickness=-1)

    return vis_map, simple_map

def add_goal(vis_map, simple_map):
    traversable = np.where(simple_map == 1)
    num_points = len(traversable[0])
    if num_points > 0:
        goal_idx = random.randint(0, num_points-1)
        color = (255, 0, 0)
        goal_x = traversable[0][goal_idx]
        goal_y = traversable[1][goal_idx]
        cv2.circle(vis_map, (goal_y, goal_x), 2, color, thickness=-1)
        simple_map[goal_y, goal_x] = 3

    return vis_map, simple_map

if __name__ == "__main__":
    parser = ArgumentParser("An example showing how to generate maps with PyOblige.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    parser.add_argument("-s", "--seed",
                        default=DEFAULT_SEED,
                        type=int,
                        help="Number of iterations(actions) to run")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Use verbose mode during map generation.")
    parser.add_argument("-o", "--output_file",
                        default=DEFAULT_OUTPUT_FILE,
                        help="Where the wad file will be created.")
    parser.add_argument("-x", "--exit",
                        action="store_true",
                        help="Do not test the wad, just leave after generation.")

    args = parser.parse_args()

    game = vzd.DoomGame()
    # Use your config
    game.load_config(args.config)
    game.set_doom_map("map01")
    game.set_doom_skill(3)

    # Create Doom Level Generator instance and set optional seed.
    generator = oblige.DoomLevelGenerator()
    generator.set_seed(args.seed)

    # Set generator configs, specified keys will be overwritten.
    generator.set_config({
        "size": "regular",
        "health": "more",
        "weapons": "sooner",
        "theme": "jumble",
        "mons": "none",
        "stealth_mons": 0,
        "switches": "none",
        "teleporters": "none",
        "darkness": "none",
        "keys": "none"})

    # There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
    #generator.set_config(oblige.childs_play_wad)

    # Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
    generator.set_config({"length": "few"})

    # Generate method will return number of maps inside wad file.
    wad_path = args.output_file
    print("Generating {} ...".format(wad_path))
    num_maps = generator.generate(wad_path, verbose=args.verbose)
    print("Generated {} maps.".format(num_maps))

    if args.exit:
        exit(0)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(args.output_file)

    # Sets up game for spectator (you)
    #game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
    #game.set_window_visible(True)
    game.set_window_visible(False)
    #game.set_mode(vzd.Mode.SPECTATOR)
    game.set_render_hud(False)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables rendering of automap.
    game.set_automap_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)

    # All map's geometry and objects will be displayed.
    game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.add_available_game_variable(vzd.GameVariable.PITCH)
    game.add_available_game_variable(vzd.GameVariable.ROLL)

    game.add_available_game_variable(vzd.GameVariable.VELOCITY_X)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Y)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Z)

    # This CVAR can be used to make a map follow a player.
    game.add_game_args("+am_followplayer 1")

    # This CVAR controls scale of rendered map (higher valuer means bigger zoom).
    game.add_game_args("+viz_am_scale 3")

    # This CVAR shows the whole map centered (overrides am_followplayer and viz_am_scale).
    #game.add_game_args("+viz_am_center 1")

    # Map's colors can be changed using CVARs, full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
    #game.add_game_args("+am_backcolor 000000")

    game.add_game_args("+am_showthingsprites 3")
    game.add_game_args("+am_cheat 1")

    game.init()

    # Play as many episodes as maps in the new generated WAD file.
    #episodes = num_maps
    episodes = 1

    # Play until the game (episode) is over.
    actions_num = game.get_available_buttons_size()

    map_out = cv2.VideoWriter('map_vis.avi',
                               cv2.VideoWriter_fourcc(*'X264'),
                               vzd.DEFAULT_TICRATE, (513, 513))

    vid_out = cv2.VideoWriter('screen_vis.avi',
                               cv2.VideoWriter_fourcc(*'X264'),
                               vzd.DEFAULT_TICRATE, (1280, 960))

    for i in range(1, episodes + 1):

        # Update map name
        print("Map {}/{}".format(i, episodes))
        map = "map{:02}".format(i)
        game.set_doom_map(map)
        game.new_episode()

        time = 0

        # Sleep time between actions in ms
        sleep_time = 10
        step = 0

        while not game.is_episode_finished():
            state = game.get_state()

            # Shows automap buffer
            auto_map_buffer = state.automap_buffer
            depth_buffer = state.depth_buffer
            labels_buffer = state.labels_buffer
            screen_buffer = state.screen_buffer

            time = game.get_episode_time()

            vis_map, simple_map = compute_map(state)
            vis_map, simple_map = add_goal(vis_map, simple_map)

            ret = map_out.write(vis_map)
            ret = vid_out.write(screen_buffer)

            action = spin_agent(actions_num)
            reward = game.make_action(action)
            last_action = game.get_last_action()

            #if auto_map_buffer is not None:
            #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
            #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
            #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

            cv2.waitKey(sleep_time)

            step = step + 1

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("Kills:", game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        print("Items:", game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
        print("Secrets:", game.get_game_variable(vzd.GameVariable.SECRETCOUNT))
        print("Time:", time / 35, "s")
        print("************************")
        sleep(2.0)

    game.close()
    map_out.release()
    vid_out.release()

    # Remove output of generator
    os.remove(wad_path)
    os.remove(wad_path.replace("wad", "old"))
    os.remove(wad_path.replace("wad", "txt"))
