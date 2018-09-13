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
import cv2
import numpy as np
import math

DEFAULT_CONFIG = "../../scenarios/probe.cfg"

def spin_agent(actions_num):
    actions = [ 0.0 ] * actions_num
    actions[8] = 1.0
    return actions

def path_plan(actions_num, simple_map, vis_map):
    action_sequence = []

    no_beacon = np.sum(simple_map == 4) == 0
    if no_beacon:
        actions = [ 0.0 ] * actions_num
        actions[8] = 1.0
        return [actions], vis_map

    center = simple_map.shape[0]/2.0

    beacon_locs = np.where(simple_map == 4)
    min_dist = 9999

    nx = center
    ny = center

    for l in zip(beacon_locs[0], beacon_locs[1]):
        xdiff = l[0] - center
        ydiff = l[1] - center
        d = abs(xdiff) + abs(ydiff)
        if (d < min_dist):
            min_dist = d

            nx = l[0]
            ny = l[1]

    cx = int(center)
    cy = int(center)

    while(abs(cx-nx) > 0 or abs(cy-ny)>0):
        actions = [ 0.0 ] * actions_num
        change = False
        if (cx - nx > 0) and (simple_map[cx - 1, cy] > 0):
            cx = cx - 1
            actions[7] = 1.0
            action_sequence.append(actions)
            change = True
        elif (cy - ny < 0) and (simple_map[cx, cy + 1] > 0):
            cy = cy + 1
            actions[4] = 1.0
            action_sequence.append(actions)
            change = True
        elif (cy - ny > 0) and (simple_map[cx, cy - 1] > 0):
            cy = cy - 1
            actions[5] = 1.0
            action_sequence.append(actions)
            change = True

        if not change:
            break

        vis_map[cx, cy] = (0, 255, 255)

    return action_sequence, vis_map

def spin_beeline_agent(actions_num, simple_map):
    actions = [ 0.0 ] * actions_num
    no_beacon = np.sum(simple_map == 4) == 0
    if no_beacon:
        actions[8] = 1.0
        return actions

    center = simple_map.shape[0]/2.0

    beacon_locs = np.where(simple_map == 4)
    min_dist = 9999

    nx = center
    ny = center

    for l in zip(beacon_locs[0], beacon_locs[1]):
        xdiff = l[0] - center
        ydiff = l[1] - center
        d = abs(xdiff) + abs(ydiff)
        if (d < min_dist):
            min_dist = d
            actions[7] = 1.0
            if (abs(ydiff) > 3):
                if (ydiff < 0):
                    actions[9] = 1.0
                else:
                    actions[8] = 1.0

        nx = l[0]
        ny = l[1]

    return actions

def random_probe_sequence(length, actions_num):
    action_sequence = []
    for i in range(length):
        actions = [ 0.0 ] * actions_num
        action_idx = random.choice([4, 5, 7, 8, 9])
        actions[action_idx] = 1.0
        action_sequence.append(actions)

    return action_sequence

def compute_map(state, height=960, width=1280,
               map_size=256, map_scale=3, fov=90.0,
               beacon_scale=100, pick_new_goal=False,
               only_visible_beacons=True, curr_goal=None):

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

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            color = (0, 0, 255)
            object_id = 2
            simple_map[rotated_x, rotated_y] = object_id
            cv2.circle(vis_map, (rotated_y, rotated_x), 2, color, thickness=-1)

    quantized_x = int(player_x/beacon_scale) * beacon_scale
    quantized_y = int(player_y/beacon_scale) * beacon_scale
    beacon_radius = 10

    beacons = []
    for bnx in range(-beacon_radius, beacon_radius+1):
        for bny in range(-beacon_radius, beacon_radius+1):
            beacon_x = quantized_x + bnx * beacon_scale
            beacon_y = quantized_y + bny * beacon_scale
            beacons.append((beacon_x, beacon_y))

    visble_beacons_world = []

    for b in beacons:
        object_relative_x = -b[0] + player_x
        object_relative_y = -b[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            color = (255, 0, 0)
            object_id = 3
            show = True
            if simple_map[rotated_x, rotated_y] == 0:
                show = (only_visible_beacons != True)
            else:
                visble_beacons_world.append((b[0], b[1]))

            if show:
                simple_map[rotated_x, rotated_y] = object_id
                cv2.circle(vis_map, (rotated_y, rotated_x), 2, color, thickness=-1)

    if pick_new_goal:
        if len(visble_beacons_world) > 0:
            beacon_idx = random.randint(0, len(visble_beacons_world)-1)
            """
            max_dist = 0
            for b in range(len(visble_beacons_world)):
                xdiff = visble_beacons_world[b][0] - player_x
                ydiff = visble_beacons_world[b][1] - player_y
                d = abs(xdiff) + abs(ydiff)
                if d > max_dist:
                    beacon_idx = b
                    max_dist = d
            """

            curr_goal = visble_beacons_world[beacon_idx]

    if curr_goal is not None:
        object_relative_x = -curr_goal[0] + player_x
        object_relative_y = -curr_goal[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            color = (255, 255, 0)
            object_id = 4
            if simple_map[rotated_x, rotated_y] > 0:
                simple_map[rotated_x, rotated_y] = object_id
                cv2.circle(vis_map, (rotated_y, rotated_x), 2, color, thickness=-1)

    return vis_map, simple_map, curr_goal

if __name__ == "__main__":
    parser = ArgumentParser("An example showing how to generate maps with PyOblige.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()
    # Use your config
    game.load_config(args.config)
    game.set_doom_map("map01")
    game.set_doom_skill(3)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path('gen_scene_13.wad')

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

    map_out = cv2.VideoWriter('map_vis_probe' + '.avi',
                               cv2.VideoWriter_fourcc(*'X264'),
                               vzd.DEFAULT_TICRATE, (513, 513))

    vid_out = cv2.VideoWriter('screen_vis_probe' + '.avi',
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
        curr_goal = None

        action_sequence = []

        while not game.is_episode_finished():
            state = game.get_state()

            # Shows automap buffer
            auto_map_buffer = state.automap_buffer
            depth_buffer = state.depth_buffer
            labels_buffer = state.labels_buffer
            screen_buffer = state.screen_buffer

            time = game.get_episode_time()

            pick_new_goal = False
            if len(action_sequence) == 0:
                pick_new_goal = True

            vis_map, simple_map, curr_goal = compute_map(state,
                                                         pick_new_goal = pick_new_goal,
                                                         curr_goal = curr_goal)
            #vis_map, simple_map = add_goal(vis_map, simple_map)

            ret = map_out.write(vis_map)
            ret = vid_out.write(screen_buffer)

            action = spin_beeline_agent(actions_num, simple_map)

            reward = game.make_action(action)
            last_action = game.get_last_action()

            #cv2.imshow('Simple map', vis_map)
            #cv2.imshow('Plan', vis_plan)
            #cv2.imshow('Screen', screen_buffer)

            if pick_new_goal:
                cv2.imwrite('data/map_' + str(step) + '.png', vis_map)
                #cv2.waitKey(0)


            #if auto_map_buffer is not None:
            #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
            #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
            #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

            cv2.waitKey(sleep_time)

            step = step + 1
            print('step:', step)

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
