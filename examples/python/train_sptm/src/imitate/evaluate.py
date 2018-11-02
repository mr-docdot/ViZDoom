import numpy as np
import vizdoom as vzd

from keras.metrics import binary_accuracy
from keras.models import load_model
from setup import setup_game_test


def evaluate_scenario(model, wad_path, lmp_path, goals_path):
    num_steps = 4200
    history_size = 2
    rec_goals = np.load(goals_path)
    accs = []

    # Setup game using WAD and LMP
    game = setup_game_test(wad_path)
    game.replay_episode(lmp_path)

    # Declare state history matrices
    frames = np.zeros((history_size + num_steps, 240, 320, 3))
    depths = np.zeros((history_size + num_steps, 240, 320))
    angles = np.zeros((history_size + num_steps, 1))
    goals = np.zeros((history_size + num_steps, 2))
    # actions = np.zeros((history_size + num_steps, 21))

    for step in range(num_steps - 1):
        current_time = game.get_episode_time()

        # Get state from game
        state = game.get_state()
        frame = state.screen_buffer
        depth = state.depth_buffer
        angle = game.get_game_variable(vzd.GameVariable.ANGLE)

        # Record state
        data_idx = step + history_size
        frames[data_idx] = frame
        depths[data_idx] = depth
        angles[data_idx] = angle
        goals[data_idx] = rec_goals[current_time - 1]

        # Get ground true action from game (FORWARD, RIGHT, LEFT)
        game.advance_action()
        last_action = game.get_last_action()
        action_batch = np.array(last_action[7:10])[np.newaxis, :]
        # actions[data_idx] = last_action

        # Build input for network
        batch_rgbd_all = []
        batch_ga_all = []

        for j in reversed(range(history_size + 1)):
            batch_frames = frames[step + j, :, :, :]
            batch_depths = depths[step + j, :, :][:, :, np.newaxis]
            batch_rgbd_all.append(batch_frames)
            batch_rgbd_all.append(batch_depths)

            batch_goals = goals[step + j, :]
            batch_angles = angles[step + j, :]
            batch_ga_all.append(batch_goals)
            batch_ga_all.append(batch_angles)

        batch_rgbd = np.concatenate(batch_rgbd_all, axis=2)[np.newaxis, :, :, :]
        batch_ga = np.concatenate(batch_ga_all, axis=0)[np.newaxis, :]

        # Compute and record test loss at iteration
        test_acc = model.test_on_batch([batch_rgbd, batch_ga], action_batch)[1] # NOQA

        # if action_batch[0, 0] == 1 and action_batch[0, 1] == 0 and action_batch[0, 2] == 0: # NOQA
        #     pass
        # elif action_batch[0, 0] == 0 and action_batch[0, 1] == 1 and action_batch[0, 2] == 0: # NOQA
        #     pass
        # elif action_batch[0, 0] == 1 and action_batch[0, 1] == 1 and action_batch[0, 2] == 0: # NOQA
        #     pass
        # else:
        #     accs.append(test_acc)
        accs.append(test_acc)

    return np.mean(np.array(accs))


model_path = '../../experiments/trained_models/model_angle_history.2500.h5'
test_wad_ids = [192, 194, 195, 196, 197, 199, 201, 202, 203, 204]
test_lmp_id = 2

model = load_model(model_path)
accs = np.zeros(len(test_wad_ids))

for idx, wad_id in enumerate(test_wad_ids):
    wad_path = '../../data/maps/out/gen_{}_size_regular_mons_none_steepness_none.wad'.format(wad_id) # NOQA
    lmp_path = '../../data/exploration/{}_{}_rec.lmp'.format(wad_id, test_lmp_id) # NOQA
    goals_path = '../../data/exploration/{}_{}_rec.npy'.format(wad_id, test_lmp_id) # NOQA

    acc = evaluate_scenario(model, wad_path, lmp_path, goals_path)
    accs[idx] = acc
    print(acc)

print('TOTAL ACCURACY: ', np.mean(accs))
