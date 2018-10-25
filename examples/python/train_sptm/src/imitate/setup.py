import os
import tensorflow as tf
import vizdoom as vzd

from keras.backend.tensorflow_backend import set_session
from os.path import exists, join

# limit memory usage
config = tf.ConfigProto()
TRAIN_MEMORY_FRACTION = 0.4
config.gpu_options.per_process_gpu_memory_fraction = TRAIN_MEMORY_FRACTION
set_session(tf.Session(config=config))

DEFAULT_CONFIG = '../explore/explorer.cfg'


def setup_training_paths(experiment_id):
    # Built appropriate paths
    experiments_dir = '../../experiments/'
    experiment_dir = join(experiments_dir, '{}/'.format(experiment_id))
    logs_path = join(experiment_dir, 'logs/')
    models_path = join(experiment_dir, 'models/')
    current_model_path = join(models_path, 'model.{epoch:02d}.h5')

    # Check that dir doesn't already exist so we don't overwrite
    assert (not exists(experiment_dir)), 'Experiment dir {} already exists'\
        .format(experiment_dir)

    # Create appropriate directories
    os.makedirs(experiment_dir)
    os.makedirs(logs_path)
    os.makedirs(models_path)

    return logs_path, current_model_path


def setup_game(wad):
    game = vzd.DoomGame()

    # Use your config
    game.load_config(DEFAULT_CONFIG)

    # Set Scenario to the new generated WAD
    game.set_doom_scenario_path(wad)

    # Sets up game for spectator (you)
    # game.add_game_args("+freelook 1")
    # game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
    game.set_window_visible(False)
    # game.set_mode(vzd.Mode.SPECTATOR)
    game.set_render_hud(False)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables rendering of automap.
    game.set_automap_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)

    # All map's geometry and objects will be displayed.
    game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)

    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.add_available_game_variable(vzd.GameVariable.PITCH)
    game.add_available_game_variable(vzd.GameVariable.ROLL)

    game.add_available_game_variable(vzd.GameVariable.VELOCITY_X)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Y)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Z)

    # This CVAR can be used to make a map follow a player.
    game.add_game_args("+am_followplayer 1")

    # This CVAR controls scale of rendered map (higher valuer means bigger zoom).
    game.add_game_args("+viz_am_scale 3")

    # This CVAR shows the whole map centered (overrides am_followplayer and viz_am_scale).
    # game.add_game_args("+viz_am_center 1")

    # Map's colors can be changed using CVARs, full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
    # game.add_game_args("+am_backcolor 000000")

    game.add_game_args("+am_showthingsprites 3")
    game.add_game_args("+am_cheat 1")
    game.add_game_args("+sv_cheats 1")

    game.init()
    return game
