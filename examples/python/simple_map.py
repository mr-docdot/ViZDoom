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
import math

DEFAULT_CONFIG = "../../scenarios/oblige.cfg"
DEFAULT_SEED = 999
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
        "mons": "more"})

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
    episodes = num_maps
    simple_map = np.zeros((257, 257), dtype=np.uint8)


    # Play until the game (episode) is over.
    for i in range(1, episodes + 1):

        # Update map name
        print("Map {}/{}".format(i, episodes))
        map = "map{:02}".format(i)
        game.set_doom_map(map)
        game.new_episode()

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

            player_x = state.game_variables[5]
            player_y = state.game_variables[6]
            player_z = state.game_variables[7]

            player_angle = state.game_variables[8]
            player_pitch = state.game_variables[9]
            player_roll = state.game_variables[10]

            print('player:', player_x, player_y, player_z)
            print('player:', player_angle, player_pitch, player_roll)

            simple_map = np.zeros((257, 257), dtype=np.uint8)

            for l in state.labels:
                object_relative_x = -l.object_position_x + player_x
                object_relative_y = -l.object_position_y + player_y
                object_relative_z = -l.object_position_z + player_z

                print(l.object_name, (object_relative_x),
                      (object_relative_y),
                      (object_relative_z))


                scaled_x = int(object_relative_x/10 + 128)
                scaled_y = int(object_relative_y/10 + 128)

                print(scaled_x, scaled_y)

                if (scaled_x >= 0 and scaled_x <= 256 and scaled_y >= 0 and scaled_y <= 256):
                    simple_map[(scaled_x, scaled_y)] = 255

                r = 128
                fov = 90

                x1 = int(r * math.cos(math.radians(player_angle)))
                y1 = int(r * math.sin(math.radians(player_angle)))

                x2 = int(r * math.cos(math.radians(player_angle + fov)))
                y2 = int(r * math.sin(math.radians(player_angle + fov)))

                _, p1, p2 = cv2.clipLine((0, 0, 257, 257), (128, 128), (128 + x1, 128 + y1))
                _, p3, p4 = cv2.clipLine((0, 0, 257, 257), (128, 128), (128 + x2, 128 + y2))

                cv2.line(simple_map, p1, p2, (255, 255, 255), thickness=1)
                cv2.line(simple_map, p3, p4, (255, 255, 255), thickness=1)

            if auto_map_buffer is not None:
                cv2.imshow('ViZDoom Simplemap Buffer', simple_map)
            #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
            #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
            #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

            episode_dict[step] = [screen_buffer, depth_buffer, auto_map_buffer, labels_buffer, state.labels,
                                  state.game_variables, last_action]

            cv2.waitKey(sleep_time)

            step = step + 1

        np.save("replay_saved_%d_%d.npy" %(DEFAULT_SEED, i), episode_dict)
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
