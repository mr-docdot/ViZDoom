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
import queue

from grid_map import update_grid_map, map_scenario, get_auto_map, \
                     filter_graph, plot_points, plot_edges
from explorer import spin_beeline_agent

def compute_map(state, height=960, width=1280,
                map_size=256, map_scale=3, fov=90.0,
                beacon_scale=50, path = []):

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

    _, p1, p2 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x1, map_size + y1))
    _, p3, p4 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x2, map_size + y2))

    game_unit = 100.0/14
    ray_cast = (depth_buffer[height/2] * game_unit)/float(map_scale)

    ray_points = [ (map_size, map_size) ]
    for i in range(canvas_size):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size),
                               (map_size, map_size),
                               (ray_y, ray_x))
        ray_points.append(p)

    cv2.fillPoly(vis_map, np.array([ray_points], dtype=np.int32), (255, 255, 255))
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

    visble_path_locations = []
    farthest_path_loc = None

    min_dist_world = float("inf")
    min_dist_path = float("inf")

    for p, d in path:
        object_relative_x = -p[0] + player_x
        object_relative_y = -p[1] + player_y
        if (abs(object_relative_y) + abs(object_relative_x)) < min_dist_world:
            min_dist_world = abs(object_relative_y) + abs(object_relative_x)
            min_dist_path = d

    max_dist = min_dist_path

    for p, d in path:
        object_relative_x = -p[0] + player_x
        object_relative_y = -p[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
            rotated_y >= 0 and rotated_y < canvas_size):
            color = (255, 0, 0)
            object_id = 3
            show = False
            if simple_map[rotated_x, rotated_y] > 0:
                visble_path_locations.append((p[0], p[1]))
                if d > max_dist:
                    max_dist = d
                    farthest_path_loc = (p[0], p[1])

            if show:
                simple_map[rotated_x, rotated_y] = object_id
                cv2.circle(vis_map, (rotated_y, rotated_x),
                           2, color, thickness=-1)

    if farthest_path_loc is not None:
        object_relative_x = -farthest_path_loc[0] + player_x
        object_relative_y = -farthest_path_loc[1] + player_y

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

    return vis_map, simple_map, farthest_path_loc

def bfs(start, nodes, edges, grid_scale = 50):
    in_queue = {}
    distances = {}
    s = queue.Queue()
    s.put((start, 0, None))
    in_queue[start] = True

    while not s.empty():
        curr, d, prev = s.get()
        distances[curr] = (d, prev)
        neighbors = [ (curr[0], curr[1] - grid_scale),
                      (curr[0], curr[1] + grid_scale),
                      (curr[0] - grid_scale, curr[1]),
                      (curr[0] + grid_scale, curr[1]),
                      (curr[0] - grid_scale, curr[1] - grid_scale),
                      (curr[0] + grid_scale, curr[1] + grid_scale),
                      (curr[0] - grid_scale, curr[1] + grid_scale),
                      (curr[0] + grid_scale, curr[1] - grid_scale),
                    ]

        for n in neighbors:
            if (n not in in_queue) and ((curr, n) in edges):
                s.put((n, d+1, curr))
                in_queue[n] = True

    return distances

def pick_path(curr_location, nodes, edges, min_path_length = 15):
    nearest_node = None
    min_dist = float("inf")
    for n in nodes:
        dist = abs(curr_location[0] - n[0]) + abs(curr_location[1] - n[1])
        if dist < min_dist:
            min_dist = dist
            nearest_node = n

    distances = bfs(nearest_node, nodes, edges)

    distant_nodes = []
    for n in distances:
        dist, _ = distances[n]
        if dist >= min_path_length:
            distant_nodes.append(n)

    path = []
    distance_field = {}
    if len(distant_nodes) > 0 :
        node_idx = random.randint(0, len(distant_nodes) - 1)
        curr = distant_nodes[node_idx]
        while curr is not None:
            d, prev = distances[curr]
            path.append((curr, d))
            curr = prev

        distance_field = bfs(distant_nodes[node_idx], nodes, edges)

    return path, distance_field

def navigator(config, scenario, nodes, edges, map_vis=None,
              vid_out_name=None, min_path_length = 15):

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

    game.new_episode()
    game.send_game_command("iddqd")

    step = 0
    curr_goal = None
    explored_goals = {}

    vid_out = None
    if vid_out_name is not None:
        vid_out = cv2.VideoWriter(vid_out_name,
                                  cv2.VideoWriter_fourcc(*'X264'),
                                  vzd.DEFAULT_TICRATE, (2*1280, 960))

    path = []
    beacons = []
    trail_info = []
    num_trails = 0
    success_trails = 0

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
            start_x = player_x
            start_y = player_y

        if step%500 == 0:
            if step > 0 and path:
                num_trails = num_trails + 1
                prev_goal, _ = path[0]
                distance_to_goal = abs(prev_goal[0] - player_x) + \
                                   abs(prev_goal[1] - player_y)
                print('Distance to Goal:', distance_to_goal, len(path))
                trail_info.append(((player_x, player_y), path))
                if distance_to_goal < 100:
                    success_trails = success_trails + 1

            path, distance_field = pick_path((player_x, player_y, player_z),
                                              nodes, edges,
                                              min_path_length=min_path_length)
            birds_eye_vis = map_vis.copy()

            beacons = []
            if path:
                end, _ = path[0]
                start, _ = path[-1]

                vis_path = [ p[0] for p in path ]
                plot_points(vis_path, birds_eye_vis, start_x, start_y,
                            vis_color = (255, 0, 255), point_size = 4)

                plot_points([start], birds_eye_vis, start_x, start_y,
                            vis_color = (255, 255, 0), point_size = 4)
                plot_points([end], birds_eye_vis, start_x, start_y,
                            vis_color = (0, 255, 255), point_size = 4)

                for b in distance_field:
                    beacons.append((b, -distance_field[b][0]))

        vis_map, simple_map, curr_goal = compute_map(state,
                                                     path = beacons)
        plot_points([(player_x, player_y)], birds_eye_vis, start_x, start_y,
                    vis_color = (0, 0, 255), point_size = 2)

        action = spin_beeline_agent(actions_num, simple_map)
        reward = game.make_action(action)
        last_action = game.get_last_action()

        if vid_out:
            vis_buffer = np.concatenate([screen_buffer, birds_eye_vis], axis = 1)
            ret = vid_out.write(vis_buffer)

        #if auto_map_buffer is not None:
        #    cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
        #    cv2.imshow('ViZDoom depth Buffer', depth_buffer)
        #    cv2.imshow('ViZDoom labels Buffer', labels_buffer)

        step = step + 1

    game.close()
    if vid_out:
        vid_out.release()

    return trail_info, num_trails, success_trails

if __name__ == "__main__":
    scenario_path = 'gen_1_size_regular_mons_none_steepness_none.wad'
    config_path = '../../../scenarios/explorer.cfg'
    auto_map = get_auto_map(config_path, scenario_path)

    paths = []

    nodes = {}
    edges = {}
    for e in range(4):
        start_point = map_scenario(config_path, scenario_path,
                                   e, nodes = nodes, edges = edges)

    filter_graph(nodes, edges)
    plot_points(nodes.keys(), auto_map, start_point[0], start_point[1])
    plot_edges(edges.keys(), auto_map, start_point[0], start_point[1])

    info, num_trails, success_trails = \
            navigator(config_path, scenario_path,
                      nodes, edges, map_vis = auto_map,
                      vid_out_name = 'test_navigator.avi',
                      min_path_length = 30)

    np.save('test.npy', np.array([info, num_trails, success_trails]))
    print(success_trails, num_trails)
