from __future__ import print_function

import os
from random import choice
import vizdoom as vzd
import cv2

game = vzd.DoomGame()

# Set Scenario to the new generated WAD
game.set_doom_scenario_path('gen_scene_111.wad')

# Sets up game for spectator (you)
game.add_game_args("+freelook 1")
game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
game.set_window_visible(True)
game.set_mode(vzd.Mode.SPECTATOR)
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
game.add_game_args("+viz_am_scale 5")

# This CVAR shows the whole map centered (overrides am_followplayer and viz_am_scale).
#game.add_game_args("+viz_am_center 1")

# Map's colors can be changed using CVARs, full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
#game.add_game_args("+am_backcolor 000000")

game.add_game_args("+am_showthingsprites 3")
game.add_game_args("+am_cheat 0")
game.add_game_args("+r_particles 0")
game.add_game_args("+r_drawtrans 0")
game.add_game_args("+cl_maxdecals 0")

game.init()

game.replay_episode('./episode_111_4_rec.lmp')

sleep_time = 10

while not game.is_episode_finished():
    s = game.get_state()

    auto_map_buffer = s.automap_buffer
    depth_buffer = s.depth_buffer
    labels_buffer = s.labels_buffer
    screen_buffer = s.screen_buffer

    # Use advance_action instead of make_action.
    game.advance_action()

    if auto_map_buffer is not None:
        cv2.imshow('ViZDoom Automap Buffer', auto_map_buffer)
        cv2.imshow('ViZDoom depth Buffer', depth_buffer)
        cv2.imshow('ViZDoom labels Buffer', labels_buffer)

    cv2.waitKey(sleep_time)

    r = game.get_last_reward()
    # game.get_last_action is not supported and don't work for replay at the moment.

    print("State #" + str(s.number))
    print("Game variables:", s.game_variables[0])
    print("Reward:", r)
    print("=====================")

print("Episode finished.")
print("total reward:", game.get_total_reward())
print("************************")

game.close()
