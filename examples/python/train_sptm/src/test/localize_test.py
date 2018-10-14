import cv2
import grid_map
import matplotlib.pyplot as plt
import networkx as nx
import os
import vizdoom as vzd

from build_graph import main_visualize_shortcuts
from navigator import Navigator
from PIL import Image
from sptm import SPTM
from test_navigation_quantitative import *

environment = 'random1'
mode = 'policy'
exploration_model_directory = 'none'
wad_path = '../../data/Test/random1/random1.wad'
cfg_path = '../../data/maps/explorer.cfg'

# Extract keyframes from recorded exploration and build SPTM
keyframes, keyframe_coordinates, _ = main_exploration(None, environment)
memory = SPTM()
memory.set_shortcuts_cache_file(environment)
memory.last_nn = None

# Compute shortcuts and build graph from demonstration
memory.compute_shortcuts(keyframes, keyframe_coordinates)
memory.build_graph(keyframes, keyframe_coordinates)

# Pass in RGB view from current location here
test_wad = TEST_SETUPS[environment].wad
game = test_setup(test_wad)
current_screen = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)

# Perform nearest neighbors search and get closes nodes
nn, probabilities = memory.find_smoothed_nn(current_screen)

# Get start point in scenario
new_game = vzd.DoomGame()
new_game.load_config(cfg_path)
new_game.set_doom_scenario_path(wad_path)
new_game.add_available_game_variable(vzd.GameVariable.POSITION_X)
new_game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
new_game.init()

start_x = new_game.get_state().game_variables[5]
start_y = new_game.get_state().game_variables[6]

# Plot nodes on graph
node_coordinates = np.array(keyframe_coordinates)
nodes_xy = node_coordinates[:, 0:2]
auto_map = grid_map.get_auto_map(cfg_path, wad_path)
grid_map.plot_points(nodes_xy, auto_map, start_x, start_y)


# Plot edges on graph
graph = memory.graph
graph_edges = np.array(graph.edges())
shortcuts = np.where(graph_edges[:, 0] != graph_edges[:, 1] - 1)
shortcut_edges = graph_edges[shortcuts][:300]
# print(shortcut_edges)

for idx in range(100):
    pair = shortcut_edges[idx]
    img_l = Image.open('./keyframe_views/{}.png'.format(pair[0]))
    img_r = Image.open('./keyframe_views/{}.png'.format(pair[1]))
    width, height = img_l.size
    width = width * 2
    new_im = Image.new('RGB', (width, height))

    x_offset = 0
    new_im.paste(img_l, (0, 0))
    new_im.paste(img_r, (width / 2, 0))

    new_im.save('./shortcut_comparisons/{}.png'.format(idx)) 

sources = shortcut_edges[:, 0]
sinks = shortcut_edges[:, 1]
sources_xy = node_coordinates[sources, 0:2]
sinks_xy = node_coordinates[sinks, 0:2]

xy_dist = np.linalg.norm(np.abs(sources_xy - sinks_xy), axis=1)
print(xy_dist)
print(len(np.where(xy_dist > 500)[0]))
print(shortcut_edges.shape)

edges = np.stack((sources_xy, sinks_xy), axis=0)
edges = np.swapaxes(edges, 0, 1).tolist()

grid_map.plot_edges(edges, auto_map, start_x, start_y)

cv2.imwrite('./test.png', auto_map)

# Show comparisons between shortcut connections
