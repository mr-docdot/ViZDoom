import keras.optimizers
import numpy as np
import resnet
import vizdoom as vzd

from keras.utils import multi_gpu_model
from os import listdir
from os.path import isfile, join
from setup import setup_game_train, setup_training_paths


def get_sorted_wad_ids(wad_dir):
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_ids = [int(f.split('_')[1]) for f in wad_files]
    wad_ids.sort()

    return wad_ids


def set_random_lmp(game, wad_id, lmp_dir, num_saved_eps):
    lmp_template = '{}_{}_rec.lmp'
    ep_idx = np.random.randint(0, num_saved_eps, 1)[0]
    lmp_filename = lmp_template.format(wad_id, ep_idx)
    lmp_path = join(lmp_dir, lmp_filename)

    game.replay_episode(lmp_path)
    return ep_idx


def get_advance_agent_state(game):
    state = game.get_state()

    # Read frame and depth map from game state
    cur_frame = state.screen_buffer
    depth_buffer = state.depth_buffer
    angle = game.get_game_variable(vzd.GameVariable.ANGLE)

    # Get action corresponding to current frame
    game.advance_action()
    action = np.array(game.get_last_action())

    return cur_frame, depth_buffer, angle, action


def data_generator(data_dir, wad_dir, batch_size):
    '''PRE-PROCESSING'''
    # Declare hyper-parameters
    wad_template = 'gen_{}_size_regular_mons_none_steepness_none.wad'
    goals_template = '{}_{}_rec.npy'
    ep_len = 4200
    num_saved_eps = 5
    num_steps = 100

    # Set up games for each wad file
    games = []
    game_ids = []
    wad_ids = get_sorted_wad_ids(wad_dir)
    game_goals = np.zeros((len(wad_ids), ep_len, 2))

    for wad_id in wad_ids:
        wad_path = join(wad_dir, wad_template.format(wad_id))
        game = setup_game_train(wad_path)
        games.append(game)
        game_ids.append(wad_id)

    '''DATA SAMPLING'''
    while True:
        # Randomly sample batch_size number of games
        frames = np.zeros((batch_size, num_steps, 240, 320, 3))
        depths = np.zeros((batch_size, num_steps, 240, 320))
        angles = np.zeros((batch_size, num_steps, 1))
        goals = np.zeros((batch_size, num_steps, 2))
        actions = np.zeros((batch_size, num_steps, 21))

        for i, idx in enumerate(np.random.randint(0, len(games), batch_size)): # NOQA
            game = games[idx]
            game_id = game_ids[idx]

            # Set random LMP for game if never sampled before
            if game.is_new_episode():
                lmp_id = set_random_lmp(game, game_id, data_dir, num_saved_eps)
                goals_path = join(data_dir, goals_template.format(game_id, lmp_id))
                game_goals[idx] = np.load(goals_path)

            # Re-initialize game with random LMP if episode over
            if game.is_episode_finished():
                game.close()
                game.init()
                lmp_id = set_random_lmp(game, game_ids[idx], data_dir, num_saved_eps)
                goals_path = join(data_dir, goals_template.format(game_id, lmp_id))
                game_goals[idx] = np.load(goals_path)

            # Sample num_steps actions from game and record state
            for step in range(num_steps):
                current_time = game.get_episode_time()
                if not game.is_episode_finished():
                    frame, depth, angle, action = get_advance_agent_state(game)
                    frames[i][step] = frame
                    depths[i][step] = depth
                    angles[i][step] = angle
                    goals[i][step] = game_goals[idx][current_time - 1]
                    actions[i][step] = action
                else:
                    frames[i][step] = frames[i][step - 1]
                    depths[i][step] = depths[i][step - 1]
                    angles[i][step] = angles[i][step - 1]
                    goals[i][step] = goals[i][step - 1]
                    actions[i][step] = actions[i][step - 1]

        # Reshape output for batch
        for i in range(num_steps):
            batch_frames = frames[:, i, :, :, :]
            batch_depths = depths[:, i, :, :][:, :, :, np.newaxis]
            batch_rgbd = np.concatenate((batch_frames, batch_depths), axis=3)

            batch_goals = goals[:, i, :]
            batch_angles = angles[:, i, :]
            batch_GAs = np.concatenate((batch_goals, batch_angles), axis=1)

            # Predict only values in the action space (7, 8, 9)
            batch_target = actions[:, i, :]
            batch_target = batch_target[:, 7:10]
            yield ([batch_rgbd, batch_GAs], batch_target)


wad_dir = '../../data/maps/out/'
data_dir = '../../data/exploration/'
batch_size = 32

generator = data_generator(data_dir, wad_dir, batch_size)
i = 0

# Setup directories to save logs and models
experiment_id = 'beeline'
logs_path, current_model_path = setup_training_paths(experiment_id)

# Build model
model = resnet.ResnetBuilder.build_resnet_18((4, 240, 320), 3,
                                             is_classification=True)
adam = keras.optimizers.Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                             decay=0.0)
callbacks_list = [keras.callbacks.TensorBoard(log_dir=logs_path,
                                              write_graph=False),
                  keras.callbacks.ModelCheckpoint(current_model_path,
                                                  period=100)]

# Run model on multiple GPUs if available
try:
    model = multi_gpu_model(model)
    print("Training model on multiple GPUs")
except ValueError:
    print("Training model on single GPU")

# Train model
model.compile(loss='binary_crossentropy', optimizer=adam,
              metrics=['binary_accuracy'])
model.fit_generator(generator,
                    steps_per_epoch=100,
                    epochs=2500,
                    callbacks=callbacks_list)
