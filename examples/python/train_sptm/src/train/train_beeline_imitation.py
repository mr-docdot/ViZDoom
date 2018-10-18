from train_setup import setup_game

def get_advance_agent_state(wad, lmp):
  # Setup game from WAD and LMP data
  game = setup_game(wad)
  game.replay_episode(lmp)

  state = game.get_state()
  depth_buffer = state.depth_buffer
  print(depth_buffer)
  current_frame = state.screen_buffer.transpose([1, 2, 0])
  print(current_frame)

  game.advance_action()

wad_path = '../../data/maps/out/gen_1_size_regular_mons_none_steepness_none.wad'
lmp_path = '../../data/exploration/1_0_rec.lmp'
get_advance_agent_state(wad_path, lmp_path)
