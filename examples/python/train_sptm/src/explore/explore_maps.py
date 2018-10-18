import time

from beeline_navigator import explore
from natsort import natsorted
from os import listdir
from os.path import isfile, join


def get_valid_wad_paths(wad_dir):
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_paths = [join(wad_dir, f) for f in wad_files]
    wad_paths = natsorted(wad_paths)

    return wad_paths


default_cfg_path = './explorer.cfg'
wad_dir = '../../data/maps/out'
wad_paths = get_valid_wad_paths(wad_dir)
num_explorations = 5

for idx, wad_path in enumerate(wad_paths):
    wad_id = wad_path.split('/')[-1].split('_')[1]
    start = time.time()

    for i in range(num_explorations):
        explore(default_cfg_path, wad_path, i,
                lmp_out_dir='../../data/exploration')

    end = time.time()
    elapsed_time = end - start
    print('Finished exploring map {} for {} times in {}s'.format(
        idx, num_explorations, elapsed_time
    ))
