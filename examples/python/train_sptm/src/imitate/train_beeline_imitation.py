from train_setup import setup_game


def get_advance_agent_state(game):
    state = game.get_state()

    # Read frame and depth map from game state
    cur_frame = state.screen_buffer.transpose([1, 2, 0])
    depth_buffer = state.depth_buffer

    # Get action corresponding to current frame
    game.advance_action()
    action = game.get_last_action()

    return cur_frame, depth_buffer, action


wad_path = '../../data/maps/out/gen_98_size_regular_mons_none_steepness_none.wad' # NOQA
lmp_path = '../../data/exploration/98_0_rec.lmp'

# Setup game from WAD and LMP data
game = setup_game(wad_path)
game.replay_episode(lmp_path)

for i in range(500):
    get_advance_agent_state(game)

print(game.get_available_buttons())