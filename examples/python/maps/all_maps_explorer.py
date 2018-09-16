from explorer import explorer
import glob

num_episodes = 4

for scenario in glob.glob('*.wad'):
    for e in range(num_episodes):
        print(scenario, e)
        explorer('../../../scenarios/explorer.cfg', scenario, e)
