from __future__ import print_function
from time import sleep
import os
import sys
sys.path.insert(0, '/data/ravi/ViZDoom/bin/python2.7/pip_package')

import itertools as it
import random
import vizdoom as vzd
from vizdoom import Button
from argparse import ArgumentParser
import oblige
import cv2
import numpy as np
import math

def visualize(config, scenario, episode, vid_out_name=None):

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
    #game.set_automap_buffer_enabled(True)
    #game.set_labels_buffer_enabled(True)
    #game.set_depth_buffer_enabled(True)

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

    game.add_game_args("+sv_cheats 1")

    scenario_name = scenario.split('.')[0]

    game.init()

    game.set_doom_map("map01")

    game.replay_episode(scenario_name + '_' + str(episode) + '_rec.lmp')
    game.send_game_command("iddqd")

    step = 0

    vid_out = None
    if vid_out_name is not None:
        vid_out = cv2.VideoWriter(vid_out_name,
                                  cv2.VideoWriter_fourcc(*'X264'),
                                  vzd.DEFAULT_TICRATE, (1280, 960))

    visited_points = []
    start_point = None

    while not game.is_episode_finished():
        state = game.get_state()

        # Shows automap buffer
        auto_map_buffer = state.automap_buffer
        depth_buffer = state.depth_buffer
        labels_buffer = state.labels_buffer
        screen_buffer = state.screen_buffer

        player_x = state.game_variables[5]
        player_y = state.game_variables[6]
        player_z = state.game_variables[7]

        if step == 0:
            start_point = (player_x, player_y, player_z)

        visited_points.append((player_x, player_y, player_z))

        game.advance_action()

        reward = game.get_last_reward()
        last_action = game.get_last_action()

        if vid_out:
            ret = vid_out.write(screen_buffer)

        #if auto_map_buffer is not None:
        #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
        #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
        #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

        step = step + 1

    game.close()
    if vid_out:
        vid_out.release()

    return visited_points, start_point

def plot_points(points, vis_map, center_x, center_y,
                map_scale = 5, vis_color = (255, 0, 0)):
    bound_x = vis_map.shape[0]
    bound_y = vis_map.shape[1]
    cv2.circle(vis_map, (bound_y/2, bound_x/2), 4, (255, 255, 255), thickness=-1)
    for p in points:
        vis_x = int((center_y - p[1])/map_scale + bound_x/2)
        vis_y = int((p[0] - center_x)/map_scale + bound_y/2)
        if (vis_x < bound_x and vis_y < bound_y and
            vis_x >=0 and vis_y >=0):
            vis_map[(vis_x, vis_y)] = vis_color

def get_auto_map(config, scenario):

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

    #game.add_game_args("+viz_am_center 1")
    game.add_game_args("+am_followplayer 1")
    game.add_game_args("+viz_am_scale 1.0")
    game.add_game_args("+am_showthingsprites 3")
    game.add_game_args("+sv_cheats 1")
    game.add_game_args("+am_cheat 1")

    game.init()

    game.set_doom_map("map01")

    game.send_game_command("iddqd")
    game.send_game_command('am_grid 1')

    game.advance_action()

    state = game.get_state()

    return state.automap_buffer

if __name__ == "__main__":
    auto_map = get_auto_map('../../../scenarios/explorer.cfg',
                            'gen_64_size_regular_mons_none_steepness_none.wad')
    print(auto_map.shape)
    cv2.imwrite('./outs/auto_map_test.png', auto_map)

    paths = []
    color = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 255)}
    for e in range(4):
        points, start_point = visualize('../../../scenarios/explorer.cfg',
                           'gen_64_size_regular_mons_none_steepness_none.wad', e)
        plot_points(points, auto_map, start_point[0], start_point[1],
                    vis_color = color[e])

    cv2.imwrite('./outs/test.png', auto_map)
