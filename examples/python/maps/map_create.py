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

DEFAULT_CONFIG = "../../../scenarios/explorer.cfg"
DEFAULT_SEED = 15
DEFAULT_OUTPUT_FILE = "gen_scene_%d.wad" % (DEFAULT_SEED)

def forward_agent(actions_num):
    actions = [ 0.0 ] * actions_num
    actions[7] = 1.0
    return actions

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
    parser.add_argument("-d", "--map_dimension",
                        default="micro",
                        help="Size of the map.")
    parser.add_argument("-m", "--monster_quantity",
                        default="none",
                        help="Quantity of monsters.")
    parser.add_argument("-z", "--height_change",
                        default="none",
                        help="Amount of change in height.")

    args = parser.parse_args()

    game = vzd.DoomGame()
    # Use your config
    game.load_config(args.config)
    game.set_doom_map("map01")
    game.set_doom_skill(3)

    # Create Doom Level Generator instance and set optional seed.
    generator = oblige.DoomLevelGenerator()
    generator.set_seed(args.seed)

    gen_params = {
        "size": "micro",
        "health": "more",
        "weapons": "sooner",
        "theme": "jumble",
        "stealth_mons": 0,
        "mons": "none",
        "switches": "none",
        "teleporters": "none",
        "darkness": "none",
        "doors": "none",
        "cages": "none",
        "steepness": "none",
        "keys": "none"}

    gen_params["size"] = args.map_dimension
    gen_params["mons"] = args.monster_quantity
    gen_params["steepness"] = args.height_change

    output_file = '_'.join(['gen', str(args.seed),
                           'size', args.map_dimension,
                           'mons', args.monster_quantity,
                           'steepness', args.height_change]) + '.wad'

    # Set generator configs, specified keys will be overwritten.
    generator.set_config(gen_params)

    # There are few predefined sets of settings already defined in Oblige package, like test_wad and childs_play_wad
    #generator.set_config(oblige.childs_play_wad)

    # Tell generator to generate few maps (options for "length": "single", "few", "episode", "game").
    generator.set_config({"length": "single"})

    # Generate method will return number of maps inside wad file.
    wad_path = output_file
    print("Generating {} ...".format(wad_path))
    num_maps = generator.generate(wad_path)
    print("Generated {} maps.".format(num_maps))

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(output_file)

    # Sets up game for spectator (you)
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
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
    game.add_game_args("+sv_cheats 1")

    game.init()

    # Play as many episodes as maps in the new generated WAD file.
    #episodes = num_maps
    episodes = 1

    actions_num = game.get_available_buttons_size()

    for i in range(1, episodes + 1):

        # Update map name
        print("Map {}/{}".format(i, episodes))
        map = "map{:02}".format(i)
        game.set_doom_map(map)

        game.new_episode()
        game.send_game_command("notarget")

        time = 0

        # Sleep time between actions in ms
        step = 0
        start_pos = None
        end_pos = None

        while not game.is_episode_finished():
            state = game.get_state()
            if step == 0:
                start_pos = (state.game_variables[5],
                             state.game_variables[6],
                             state.game_variables[7])

            # Shows automap buffer
            auto_map_buffer = state.automap_buffer
            depth_buffer = state.depth_buffer
            labels_buffer = state.labels_buffer
            screen_buffer = state.screen_buffer

            time = game.get_episode_time()

            action = forward_agent(actions_num)
            reward = game.make_action(action)
            last_action = game.get_last_action()


            step = step + 1
            if (step > 100):
                end_pos = (state.game_variables[5],
                           state.game_variables[6],
                           state.game_variables[7])
                break

        print("Episode finished!")

    cell_map = False
    if (abs(end_pos[0]-start_pos[0]) + abs(end_pos[1] - start_pos[1])
        < 100):
        print('distance', abs(end_pos[0]-start_pos[0]) + abs(end_pos[1] - start_pos[1]))
        cell_map = True


    game.close()

    # Remove output of generator
    if cell_map:
        os.remove(wad_path)
        os.remove(wad_path.replace("wad", "txt"))
        os.remove(wad_path.replace("wad", "old"))
