import random

from os import listdir
from os.path import isfile, join
from vizdoom import DoomGame, ScreenResolution


def get_sorted_wad_ids(wad_dir):
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_ids = [int(f.split('_')[1]) for f in wad_files]
    wad_ids.sort()

    return wad_ids


def setup_random_game(wad_id=None):
    # Set up VizDoom Game
    game = DoomGame()
    game.load_config("./beeline.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.set_window_visible(True)

    # Load generated map from WAD
    wad_dir = '../train_sptm/data/maps/out_val/'
    wad_ids = get_sorted_wad_ids(wad_dir)

    if wad_id is None:
        wad_id = random.choice(wad_ids)
    wad_path = '../train_sptm/data/maps/out_val/gen_{}_size_regular_mons_none_steepness_none.wad'.format(wad_id) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    game.new_episode()

    return game
