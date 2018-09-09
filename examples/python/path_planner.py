#!/usr/bin/env python3

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

import itertools as it
from random import choice
from time import sleep
import vizdoom as vzd
from argparse import ArgumentParser
import cv2
import numpy as np
import math

def spin_agent():
    return [1.0, 0.0, 0.0]

def spin_beeline_agent(simple_map):
    no_health = np.sum(simple_map == 3) == 0
    if no_health:
        return [1.0, 0.0, 0.0]

    center = simple_map.shape[0]/2.0

    med_kit_locs = np.where(simple_map == 3)
    min_dist = 9999
    right = 0.0
    left = 0.0
    forward = 0.0

    nx = center
    ny = center

    for l in zip(med_kit_locs[0], med_kit_locs[1]):
        xdiff = l[0] - center
        ydiff = l[1] - center
        d = abs(xdiff) + abs(ydiff)
        if (d < min_dist):
            min_dist = d
            forward = 1.0
            if (xdiff < 0):
                left = 1.0
            else:
                right = 1.0

        nx = l[0]
        ny = l[1]

    return [left, right, forward]

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
            if l.object_name == 'Poison':
                color = (0, 0, 255)
                object_id = 2
            else:
                color = (0, 255, 0)
                object_id = 3
            simple_map[rotated_y, rotated_x] = object_id
            cv2.circle(vis_map, (rotated_y, rotated_x), 2, color, thickness=-1)

    return vis_map, simple_map

DEFAULT_CONFIG = "../../scenarios/health_gathering_supreme.cfg"

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use SPECTATOR mode.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    args = parser.parse_args()
    game = vzd.DoomGame()

    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.

    game.load_config(args.config)

    # Enables freelook in engine
    #game.add_game_args("+freelook 1")

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
    game.set_window_visible(True)
    #game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables rendering of automap.
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.add_available_game_variable(vzd.GameVariable.PITCH)
    game.add_available_game_variable(vzd.GameVariable.ROLL)

    game.add_available_game_variable(vzd.GameVariable.VELOCITY_X)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Y)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Z)

    game.init()

    episodes = 10
    sleep_time = 20

    actions_num = game.get_available_buttons_size()
    actions = []
    for perm in it.product([False, True], repeat=actions_num):
        actions.append(list(perm))

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()

            vis_map, simple_map = compute_map(state, height=480, width=640)

            # Makes a random action and save the reward.
            #action = spin_agent()
            action = spin_beeline_agent(simple_map)
            reward = game.make_action(action)
            last_action = game.get_last_action()

            #game.advance_action()
            #reward = game.get_last_reward()

            cv2.imshow('ViZDoom Simple Map', vis_map)
            cv2.waitKey(sleep_time)

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

    game.close()
