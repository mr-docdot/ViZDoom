#!/usr/bin/env python3

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

DEFAULT_CONFIG = "./explorer.cfg"

def spin_agent(actions_num):
    actions = [ 0.0 ] * actions_num
    actions[8] = 1.0
    return actions

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

def compute_map(state, height=960, width=1280,
               map_size=256, map_scale=3, fov=90.0,
               beacon_scale=50, pick_new_goal=False,
               only_visible_beacons=True, curr_goal=None,
               explored_goals = {}):

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

    game_unit = 100.0/14
    ray_cast = (depth_buffer[height/2] * game_unit)/float(map_scale)

    ray_points = [ (map_size, map_size) ]
    for i in range(10, canvas_size-10):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size), (map_size, map_size),
                                                          (ray_y, ray_x))
        ray_points.append(p)

    cv2.fillPoly(vis_map, np.array([ray_points], dtype=np.int32), (255, 255, 255))
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

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
        unexplored_beacons = []
        for b in visble_beacons_world:
            if b not in explored_goals:
                unexplored_beacons.append(b)

        if len(unexplored_beacons) > 0:
            beacon_idx = random.randint(0, len(unexplored_beacons)-1)
            """
            max_dist = 0
            for b in range(len(unexplored_beacons)):
                xdiff = unexplored_beacons[b][0] - player_x
                ydiff = unexplored_beacons[b][1] - player_y
                d = abs(xdiff) + abs(ydiff)
                if d > max_dist:
                    beacon_idx = b
                    max_dist = d
            """
            curr_goal = unexplored_beacons[beacon_idx]
            explored_goals[curr_goal] = True
        else:
            curr_goal = None

        """
        elif len(visble_beacons_world) > 0:
            beacon_idx = random.randint(0, len(visble_beacons_world)-1)
            curr_goal = visble_beacons_world[beacon_idx]
            explored_goals[curr_goal] = True
        """

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

def explorer(config, scenario, episode, vid_out_name=None):

    game = vzd.DoomGame()
    # Use your config
    game.load_config(config)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(scenario)

    # Sets up game for spectator (you)
    #game.add_game_args("+freelook 1")
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

    scenario_name = scenario.split('.')[0]

    game.init()

    # Play until the game (episode) is over.
    actions_num = game.get_available_buttons_size()

    game.set_doom_map("map01")

    game.new_episode(scenario_name + '_' + str(episode) + '_rec.lmp')
    game.send_game_command("iddqd")

    step = 0
    curr_goal = None
    explored_goals = {}

    vid_out = None
    if vid_out_name is not None:
        vid_out = cv2.VideoWriter(vid_out_name,
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  vzd.DEFAULT_TICRATE, (2*1280, 960))

    while not game.is_episode_finished():
        state = game.get_state()

        # Shows automap buffer
        auto_map_buffer = state.automap_buffer
        depth_buffer = state.depth_buffer
        labels_buffer = state.labels_buffer
        screen_buffer = state.screen_buffer

        pick_new_goal = False
        if step%50 == 0:
            pick_new_goal = True

        vis_map, simple_map, curr_goal = compute_map(state,
                                                     pick_new_goal = pick_new_goal,
                                                     curr_goal = curr_goal,
                                                     explored_goals = explored_goals)

        action = spin_beeline_agent(actions_num, simple_map)
        reward = game.make_action(action)
        last_action = game.get_last_action()

        vis_map_large = np.zeros(screen_buffer.shape, dtype=np.uint8)
        vis_map_large[0:vis_map.shape[0], 0:vis_map.shape[1], :] = vis_map

        if vid_out:
            vis_buffer = np.concatenate([screen_buffer, vis_map_large], axis=1)
            ret = vid_out.write(vis_buffer)

        #if auto_map_buffer is not None:
        #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
        #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
        #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

        step = step + 1

    game.close()
    if vid_out:
        vid_out.release()

if __name__ == "__main__":
    explorer(DEFAULT_CONFIG, '../data/Test/random1/random1.wad', 99, vid_out_name='debug.avi')
