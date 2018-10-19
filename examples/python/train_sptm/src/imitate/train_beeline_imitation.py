import numpy as np

from os import listdir
from os.path import isfile, join
from setup import setup_game


def get_sorted_wad_ids(wad_dir):
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_ids = [int(f.split('_')[1]) for f in wad_files]
    wad_ids.sort()

    return wad_ids


def get_advance_agent_state(game):
    state = game.get_state()

    # Read frame and depth map from game state
    cur_frame = state.screen_buffer
    depth_buffer = state.depth_buffer

    # Get action corresponding to current frame
    game.advance_action()
    action = np.array(game.get_last_action())

    return cur_frame, depth_buffer, action


def data_generator(data_path, wad_path, batch_size):
    '''PRE-PROCESSING'''
    # Declare hyper-parameters
    wad_template = 'gen_{}_size_regular_mons_none_steepness_none.wad'
    lmp_template = '{}_{}_rec.lmp'
    goals_template = '{}_{}_rec.npy'
    ep_len = 4200
    num_saved_eps = 5
    num_steps = 100

    # Set up games for each wad file
    games = []
    game_ids = []
    wad_ids = get_sorted_wad_ids(wad_path)
    for wad_id in wad_ids[:1]:
        wad_path = join(wad_dir, wad_template.format(wad_id))
        game = setup_game(wad_path)
        games.append(game)
        game_ids.append(wad_id)

    '''DATA SAMPLING'''
    while True:
        # Randomly sample batch_size number of games
        frames = np.zeros((batch_size, num_steps, 240, 320, 3))
        depths = np.zeros((batch_size, num_steps, 240, 320))
        actions = np.zeros((batch_size, num_steps, 21))

        for i, idx in enumerate(np.random.randint(0, len(games), batch_size)): # NOQA
            game = games[idx]

            # Initialize game with LMP if never sampled before
            if game.is_new_episode():
                ep_idx = np.random.randint(0, num_saved_eps, 1)[0]
                lmp_filename = lmp_template.format(game_ids[idx], ep_idx)
                lmp_path = join(data_path, lmp_filename)
                game.replay_episode(lmp_path)

            # Sample num_steps actions from game and record state
            for step in range(num_steps):
                frame, depth, action = get_advance_agent_state(game)
                frames[i][step] = frame
                depths[i][step] = depth
                actions[i][step] = action

        # Reshape output for batch
        for i in range(num_steps):
            batch_frames = frames[:, i, :, :, :]
            batch_depths = depths[:, i, :, :][:, :, :, np.newaxis]
            batch_x = np.concatenate((batch_frames, batch_depths), axis=3)
            batch_y = actions[:, i, :]
            yield (batch_x, batch_y)


wad_dir = '../../data/maps/out/'
data_dir = '../../data/exploration/'

data_generator(data_dir, wad_dir, 4)
