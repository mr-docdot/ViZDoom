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

def update_grid_map(state, height=960, width=1280,
                    map_size=256, map_scale=3, fov=90.0,
                    grid_scale=50, nodes = {}, edges = {}):

    depth_buffer = state.depth_buffer

    player_x = state.game_variables[5]
    player_y = state.game_variables[6]
    player_z = state.game_variables[7]

    player_angle = state.game_variables[8]
    player_pitch = state.game_variables[9]
    player_roll = state.game_variables[10]

    canvas_size = 2*map_size + 1
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
    for i in range(canvas_size):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size), (map_size, map_size),
                                                          (ray_y, ray_x))
        ray_points.append(p)

    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

    quantized_x = int(player_x/grid_scale) * grid_scale
    quantized_y = int(player_y/grid_scale) * grid_scale
    beacon_radius = 10

    beacons = []
    for bnx in range(-beacon_radius, beacon_radius+1):
        for bny in range(-beacon_radius, beacon_radius+1):
            beacon_x = quantized_x + bnx * grid_scale
            beacon_y = quantized_y + bny * grid_scale
            beacons.append((beacon_x, beacon_y))

    visble_beacons_world = {}

    for b in beacons:
        object_relative_x = -b[0] + player_x
        object_relative_y = -b[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            if simple_map[rotated_x, rotated_y] != 0:
                visble_beacons_world[(b[0], b[1])] = True

    for b in visble_beacons_world:
        if b not in nodes:
            nodes[b] = True

        neighbors = [ (b[0], b[1] - grid_scale),
                      (b[0], b[1] + grid_scale),
                      (b[0] - grid_scale, b[1]),
                      (b[0] + grid_scale, b[1]),
                    ]

        for n in neighbors:
            if n in visble_beacons_world:
                if (b, n) not in edges:
                    edges[(b, n)] = True
                    edges[(n, b)] = True

                """
                visible_region = (simple_map > 0)
                object_relative_x1 = -n[0] + player_x
                object_relative_y1 = -n[1] + player_y

                rotated_x1 = math.cos(math.radians(-player_angle)) * object_relative_x1 - math.sin(math.radians(-player_angle)) * object_relative_y1
                rotated_y1 = math.sin(math.radians(-player_angle)) * object_relative_x1 + math.cos(math.radians(-player_angle)) * object_relative_y1

                rotated_x1 = int(rotated_x1/map_scale + map_size)
                rotated_y1 = int(rotated_y1/map_scale + map_size)

                object_relative_x2 = -b[0] + player_x
                object_relative_y2 = -b[1] + player_y

                rotated_x2 = math.cos(math.radians(-player_angle)) * object_relative_x2 - math.sin(math.radians(-player_angle)) * object_relative_y2
                rotated_y2 = math.sin(math.radians(-player_angle)) * object_relative_x2 + math.cos(math.radians(-player_angle)) * object_relative_y2

                rotated_x2 = int(rotated_x2/map_scale + map_size)
                rotated_y2 = int(rotated_y2/map_scale + map_size)

                line_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                cv2.line(line_canvas, (rotated_y1, rotated_x1), (rotated_y2, rotated_x2), (1, 1, 1))

                clipped_canvas = np.logical_and(visible_region, line_canvas > 0)

                visible_path = False
                if np.sum(line_canvas > 0) == np.sum(clipped_canvas):
                    visible_path = True

                if (b, n) not in edges and visible_path:
                    edges[(b, n)] = True
                """

def map_scenario(config, scenario, episode, nodes = {}, edges = {},
                 vid_out_name=None):

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

    start_point = None

    while not game.is_episode_finished():
        state = game.get_state()

        update_grid_map(state, nodes = nodes, edges = edges)

        player_x = state.game_variables[5]
        player_y = state.game_variables[6]
        player_z = state.game_variables[7]

        if step == 0:
            start_point = (player_x, player_y, player_z)

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

    return start_point

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

def plot_points(points, vis_map, center_x, center_y,
                map_scale = 5, vis_color = (255, 0, 0),
                point_size = 3):
    bound_x = vis_map.shape[0]
    bound_y = vis_map.shape[1]
    for p in points:
        vis_x = int((center_y - p[1])/map_scale + bound_x/2)
        vis_y = int((p[0] - center_x)/map_scale + bound_y/2)
        if (vis_x < bound_x and vis_y < bound_y and
            vis_x >=0 and vis_y >=0):
            cv2.circle(vis_map, (vis_y, vis_x), point_size, vis_color, thickness=-1)

def plot_edges(edges, vis_map, center_x, center_y,
                map_scale = 5, vis_color = (0, 255, 0)):
    bound_x = vis_map.shape[0]
    bound_y = vis_map.shape[1]
    for e in edges:
        x1 = int((center_y - e[0][1])/map_scale + bound_x/2)
        y1 = int((e[0][0] - center_x)/map_scale + bound_y/2)

        x2 = int((center_y - e[1][1])/map_scale + bound_x/2)
        y2 = int((e[1][0] - center_x)/map_scale + bound_y/2)

        if (x1 < bound_x and y1 < bound_y and x2 < bound_x and y2 < bound_y
            and x1 >=0 and y1 >=0 and x2 >=0 and y2 >=0):
            cv2.line(vis_map, (y1, x1), (y2, x2), vis_color)

def filter_graph(nodes, edges):
    nodes_with_edges = {}
    for e in edges:
        nodes_with_edges[e[0]] = True
        nodes_with_edges[e[1]] = True

    for n in nodes.keys():
        if n not in nodes_with_edges:
            nodes.pop(n)


if __name__ == "__main__":
    auto_map = get_auto_map('../../../scenarios/explorer.cfg',
                            'gen_64_size_regular_mons_none_steepness_none.wad')

    paths = []
    color = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 255)}

    nodes = {}
    edges = {}
    for e in range(1):
        start_point = map_scenario('../../../scenarios/explorer.cfg',
                                   'gen_64_size_regular_mons_none_steepness_none.wad',
                                   e, nodes = nodes, edges = edges)

    filter_graph(nodes, edges)
    plot_points(nodes.keys(), auto_map, start_point[0], start_point[1])
    plot_edges(edges.keys(), auto_map, start_point[0], start_point[1])
    cv2.imwrite('./outs/test.png', auto_map)
