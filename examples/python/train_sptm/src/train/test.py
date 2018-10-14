from os import listdir
from os.path import isfile, join

wad_dir = '../../data/maps/out/'


def get_valid_wad_ids(wad_dir):
  all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
  wad_files = [f for f in all_files if f.endswith('wad')]
  wad_ids = [int(f.split('_')[1]) for f in wad_files]
  wad_ids.sort()

  return wad_ids
