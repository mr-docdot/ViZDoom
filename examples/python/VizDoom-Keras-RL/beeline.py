import cv2
import math
import numpy as np
import random
import vizdoom as vzd


def compute_map(state, height=240, width=320,
                map_size=256, map_scale=3, fov=90.0,
                beacon_scale=50, pick_new_goal=False,
                only_visible_beacons=True, curr_goal=None,
                explored_goals={}):
    # Extract agent state from game
    depth_buffer = state.depth_buffer

    player_x = state.game_variables[5]
    player_y = state.game_variables[6]
    player_angle = state.game_variables[8]

    # Initialize maps
    canvas_size = 2*map_size + 1
    vis_map = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    simple_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Compute upper left and upper right extreme coordinates
    r = canvas_size
    offset = 225

    y1 = int(r * math.cos(math.radians(offset + player_angle)))
    x1 = int(r * math.sin(math.radians(offset + player_angle)))

    y2 = int(r * math.cos(math.radians(offset + player_angle - fov)))
    x2 = int(r * math.sin(math.radians(offset + player_angle - fov)))

    # Draw FOV boundaries
    _, p1, p2 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x1, map_size + y1))
    _, p3, p4 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x2, map_size + y2))

    # Ray cast from eye line to project depth map into 2D ray points
    game_unit = 100.0/14
    ray_cast = (depth_buffer[height/2] * game_unit)/float(map_scale)

    ray_points = [(map_size, map_size)]
    for i in range(10, canvas_size-10):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size),
                               (map_size, map_size),
                               (ray_y, ray_x))
        ray_points.append(p)

    # Fill free space on 2D map with colour
    cv2.fillPoly(vis_map, np.array([ray_points], dtype=np.int32), (255, 255, 255)) # NOQA
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

    quantized_x = int(player_x/beacon_scale) * beacon_scale
    quantized_y = int(player_y/beacon_scale) * beacon_scale
    beacon_radius = 10

    # Get beacons within range of current agent position
    beacons = []
    for bnx in range(-beacon_radius, beacon_radius+1):
        for bny in range(-beacon_radius, beacon_radius+1):
            beacon_x = quantized_x + bnx * beacon_scale
            beacon_y = quantized_y + bny * beacon_scale
            beacons.append((beacon_x, beacon_y))

    # Compute set of visible beacons and draw onto the map
    visble_beacons_world = []
    for b in beacons:
        object_relative_x = -b[0] + player_x
        object_relative_y = -b[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y # NOQA
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y # NOQA

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
           rotated_y >= 0 and rotated_y < canvas_size):
            color = (255, 0, 0)
            object_id = 3
            show = True
            if simple_map[rotated_x, rotated_y] == 0:
                show = (only_visible_beacons is not True)
            else:
                visble_beacons_world.append((b[0], b[1]))

            if show:
                simple_map[rotated_x, rotated_y] = object_id
                cv2.circle(vis_map, (rotated_y, rotated_x), 2, color,
                           thickness=-1)

    # Pick new goal from unexplored visible beacons if required
    if pick_new_goal:
        unexplored_beacons = []
        for b in visble_beacons_world:
            if b not in explored_goals:
                unexplored_beacons.append(b)

        if len(unexplored_beacons) > 0:
            beacon_idx = random.randint(0, len(unexplored_beacons)-1)
            curr_goal = unexplored_beacons[beacon_idx]
            explored_goals[curr_goal] = True
        else:
            curr_goal = None

    return curr_goal


def compute_goal(state, step, curr_goal, explored_goals):
    # Determine if new goal should be picked
    if step % 50 == 0:
        pick_new_goal = True
    else:
        pick_new_goal = False

    # Compute absolute position of goal
    curr_goal = compute_map(state,
                            pick_new_goal=pick_new_goal,
                            curr_goal=curr_goal,
                            explored_goals=explored_goals)

    # Compute relative distance to goal
    player_x = state.game_variables[5]
    player_y = state.game_variables[6]
    player_angle = state.game_variables[8]

    if curr_goal is not None:
        diff_x = curr_goal[0] - player_x
        diff_y = curr_goal[1] - player_y
        rel_goal = np.array([diff_x, diff_y])
    else:
        # Project to location behind the agent
        line_length = 250
        proj_x = player_x + math.cos(math.radians(player_angle)) * line_length * -1
        proj_y = player_y + math.sin(math.radians(player_angle)) * line_length * -1
        rel_goal = np.array([proj_x, proj_y])

    return curr_goal, rel_goal
