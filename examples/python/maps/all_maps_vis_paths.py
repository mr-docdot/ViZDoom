from visualize_paths import visualize, get_auto_map, plot_points
import glob
import cv2

for scenario in glob.glob('*steepness_heaps*.wad'):
    scenario_name = scenario.split('.')[0]
    print(scenario_name)
    size = scenario_name.split('_')[3]

    auto_map = get_auto_map('../../../scenarios/explorer.cfg', scenario)
    cv2.imwrite('./outs/%s_map.png'%(scenario_name),
                auto_map)

    paths = []
    color = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 255)}
    for e in range(4):
        points, start_point = visualize('../../../scenarios/explorer.cfg', scenario, e)
        plot_points(points, auto_map, start_point[0], start_point[1],
                    vis_color = color[e])

    cv2.imwrite('./outs/%s_vis_path.png'%(scenario_name), auto_map)
