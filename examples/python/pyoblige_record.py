#!/usr/bin/env python3

#####################################################################
# This script presents how to use the environment with PyOblige.
# https://github.com/mwydmuch/PyOblige
#####################################################################

from __future__ import print_function
from time import sleep
import os

import vizdoom as vzd
from vizdoom import Button
from argparse import ArgumentParser
import oblige
import cv2
import numpy as np

DEFAULT_CONFIG = "../../scenarios/oblige.cfg"
DEFAULT_SEED = 10
DEFAULT_OUTPUT_FILE = "gen_scene_%d.wad" % (DEFAULT_SEED)

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
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_render_hud(False)
    game.set_render_decals(False)
    game.set_render_messages(False)

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
    game.add_game_args("+am_cheat 0")

    game.add_game_args("+r_particles 0")
    game.add_game_args("+r_drawtrans 0")
    game.add_game_args("+cl_maxdecals 0")

    game.init()

    # Play as many episodes as maps in the new generated WAD file.
    episodes = num_maps


    # Play until the game (episode) is over.
    for i in range(1, episodes + 1):

        # Update map name
        print("Map {}/{}".format(i, episodes))
        map = "map{:02}".format(i)
        game.set_doom_map(map)
        game.new_episode("episode_" + str(DEFAULT_SEED) + "_" + str(i) + "_rec.lmp")

        episode_dict = {}

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

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

            #if auto_map_buffer is not None:
            #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
            #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
            #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

            episode_dict[step] = [screen_buffer, depth_buffer, auto_map_buffer, labels_buffer, state.labels,
                                  state.game_variables, last_action]

            cv2.waitKey(sleep_time)

            step = step + 1

        #np.save("replay_saved_%d_%d.npy" %(DEFAULT_SEED, i), episode_dict)
        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("Kills:", game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        print("Items:", game.get_game_variable(vzd.GameVariable.ITEMCOUNT))
        print("Secrets:", game.get_game_variable(vzd.GameVariable.SECRETCOUNT))
        print("Time:", time / 35, "s")
        print("************************")
        sleep(2.0)

    game.close()

    # Remove output of generator
    #os.remove(wad_path)
    #os.remove(wad_path.replace("wad", "old"))
    #os.remove(wad_path.replace("wad", "txt"))
