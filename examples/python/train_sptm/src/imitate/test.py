import numpy as np

from keras.models import load_model
from setup import setup_game_test


wad_path = '../../data/maps/out/gen_4_size_regular_mons_none_steepness_none.wad' # NOQA
lmp_path = '../../data/exploration/4_2_rec.lmp'
goals_path = '../../data/exploration/4_2_rec.npy'
model_path = '../../experiments/trained_models/model_binary.10.h5'

goals = np.load(goals_path)

game = setup_game_test(wad_path)
game.replay_episode(lmp_path)

goal_0 = goals[0]
model = load_model(model_path)

for i in range(100):
    state = game.get_state()
    frame = state.screen_buffer
    depth = state.depth_buffer[:, :, np.newaxis]
    rgbd = np.concatenate((frame, depth), axis=2)
    rgbd_batch = rgbd[np.newaxis, :, :, :]

    goal = goal_0
    goal_batch = goal[np.newaxis, :]
    pred_action = model.predict_on_batch([rgbd_batch, goal_batch]).tolist()[0]
    print(pred_action)

    action = [0.0] * 21
    action[7] = pred_action[0]
    action[8] = pred_action[1]
    action[9] = pred_action[2]

    game.advance_action()
    last_action = game.get_last_action()
    print(action)
    print(last_action)
    print("")


print(goal_0)
