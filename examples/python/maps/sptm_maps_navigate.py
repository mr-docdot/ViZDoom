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
import glob

from grid_map import update_grid_map, map_scenario, get_auto_map, \
                     filter_graph, plot_points, plot_edges

from navigation import navigator

for scenario_path in glob.glob('*.wad'):
    scenario_name = scenario_path.split('.')[0]
    config_path = '../../../../scenarios/explorer.cfg'
    auto_map = get_auto_map(config_path, scenario_path)
    print(scenario_name)

    paths = []

    nodes = {}
    edges = {}
    for e in range(8):
        start_point = map_scenario(config_path, scenario_path,
                                   e, nodes = nodes, edges = edges)

    filter_graph(nodes, edges)
    plot_points(nodes.keys(), auto_map, start_point[0], start_point[1])
    plot_edges(edges.keys(), auto_map, start_point[0], start_point[1])

    info, num_trails, success_trails = \
        navigator(config_path, scenario_path,
                  nodes, edges, map_vis = auto_map,
                  vid_out_name = 'navigator_%s.avi'%(scenario_name),
                  min_path_length = 15)
    np.save('navigator_%s.npy'%(scenario_name),
            np.array([info, num_trails, success_trails]))
    print(scenario_path, success_trails, num_trails)
