import numpy as np
import vizdoom as vzd

from keras.metrics import binary_accuracy
from keras.models import load_model
from setup import setup_game_test


def evaluate_scenario(model, wad_path, lmp_path, goals_path):
    steps = 4200
    goals = np.load(goals_path)
    accs = []

    # Setup game using WAD and LMP
    game = setup_game_test(wad_path)
    game.replay_episode(lmp_path)

    for i in range(steps - 1):
        # Get RGBD state from game
        state = game.get_state()
        frame = state.screen_buffer
        depth = state.depth_buffer[:, :, np.newaxis]
        angle = game.get_game_variable(vzd.GameVariable.ANGLE)

        # Build RGBD and goal input tensors
        rgbd = np.concatenate((frame, depth), axis=2)
        ga = np.append(goals[i], angle)

        rgbd_batch = rgbd[np.newaxis, :, :, :]
        ga_batch = ga[np.newaxis, :]

        # Get ground true action from game (FORWARD, RIGHT, LEFT)
        game.advance_action()
        last_action = game.get_last_action()
        action_batch = np.array(last_action[7:10])[np.newaxis, :]

        # Compute and record test loss at iteration
        test_acc = model.test_on_batch([rgbd_batch, ga_batch], action_batch)[1] # NOQA

        if action_batch[0, 0] == 1 and action_batch[0, 1] == 0 and action_batch[0, 2] == 0: # NOQA
            pass
        elif action_batch[0, 0] == 0 and action_batch[0, 1] == 1 and action_batch[0, 2] == 0: # NOQA
            pass
        elif action_batch[0, 0] == 1 and action_batch[0, 1] == 1 and action_batch[0, 2] == 0: # NOQA
            pass
        else:
            accs.append(test_acc)
        # accs.append(test_acc)

    return np.mean(np.array(accs))


model_path = '../../experiments/trained_models/model_angle.2500.h5'
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
