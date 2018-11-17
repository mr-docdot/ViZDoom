#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
from random import choice
import numpy as np

import json
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks


def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img


class A2CAgent:
    def __init__(self, state_size, action_size):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.observe = 0
        self.frame_per_action = 1

        # These are hyper parameters for the Policy Gradient
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        # Model for policy and critic network
        self.actor = None
        self.critic = None

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_num_goals = []  # Moving Average of Goals reached
        self.mavg_min_dist = []  # Moving Average of Minimum distance to goal

    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        policy = self.actor.predict(state).flatten()
        print(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print ('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + self.state_size))  # Episode_lengthx3x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[i, :, :] = self.states[i]

        # Prediction of state values for each state appears in the episode
        values = self.critic.predict(update_inputs)

        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]
        print(advantages)
        actor_loss = self.actor.fit(update_inputs, advantages, nb_epoch=1, verbose=0)
        critic_loss = self.critic.fit(update_inputs, discounted_rewards, nb_epoch=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        return actor_loss.history['loss'], critic_loss.history['loss']

    def shape_reward(self, r_t, state, prev_state, min_dist, t):
        # goal_xy = np.array([1040, -352])
        prev_xy = np.array([prev_state.game_variables[0], prev_state.game_variables[1]])
        cur_xy = np.array([state.game_variables[0], state.game_variables[1]])

        # prev_dist = np.linalg.norm(goal_xy - prev_xy)
        # cur_dist = np.linalg.norm(goal_xy - cur_xy)
        if not cur_xy[1] - prev_xy[1] == 0:
            r_t = r_t + 0.1
            print('reward')

        # if cur_dist < min_dist:
        #     r_t = r_t + 0.0005
        #     min_dist = cur_dist

        # # Check if goal has been reached
        # if r_t > 0.5:
        #     min_dist = 0

        return r_t, min_dist

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5", overwrite=True)
        self.critic.save_weights(name + "_critic.h5", overwrite=True)

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5", overwrite=True)
        self.critic.load_weights(name + "_critic.h5", overwrite=True)


if __name__ == "__main__":
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Set up VizDoom Game
    game = DoomGame()
    game.load_config("./beeline.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)

    # Load generated map from WAD
    wad_id = 194
    wad_path = '../train_sptm/data/maps/out/gen_{}_size_regular_mons_none_steepness_none.wad'.format(wad_id) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()

    # Maximum number of episodes
    max_episodes = 20000
    game.new_episode()
    game_state = game.get_state()
    action_size = game.get_available_buttons_size()

    # Set up actor and critic networks
    state_size = (4, 3)
    agent = A2CAgent(state_size, action_size)
    agent.actor = Networks.actor_network_novis(state_size, action_size, agent.actor_lr)
    agent.critic = Networks.critic_network_novis(state_size, agent.value_size, agent.critic_lr)

    # Start training
    GAME = 0
    t = 0

    # Buffer to compute rolling statistics
    num_goals_buffer, min_dist_buffer = [], []

    for i in range(max_episodes):
        game.new_episode()
        game_state = game.get_state()
        prev_state = game_state

        min_dist = 999999999

        x_t = np.array(game_state.game_variables)  # [x, y, angle]
        s_t = np.stack(([x_t]*4), axis=0)  # 4x3
        s_t = np.expand_dims(s_t, axis=0)  # 1x4x3

        steps_taken = 0  # Episode life

        while not game.is_episode_finished():
            loss = 0  # Training Loss at each update
            r_t = 0  # Initialize reward at time t
            a_t = np.zeros([action_size])  # Initialize action at time t

            # TODO: Compute beeline goal here
            # Probably should set a flag as to whether or not a new goal is needed
            # Flag should be tripped when the goal is reached, possibly in the reward

            x_t = np.array(game_state.game_variables)
            x_t = x_t.reshape(1, 1, 3)
            s_t = np.append(x_t, s_t[:, :3, :], axis=1)

            # Sample action from stochastic softmax policy
            action_idx, policy = agent.get_action(s_t)
            # print(policy)
            a_t[action_idx] = 1

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action
            game.advance_action(skiprate)

            r_t = game.get_last_reward()

            # Check if episode is terminated
            is_terminated = game.is_episode_finished()
            if not is_terminated:
                steps_taken += 1
                prev_state = game_state
                game_state = game.get_state()  # Observe again after we take the action

            # Reward Shaping
            r_t, min_dist = agent.shape_reward(r_t, game_state, prev_state, min_dist, t)

            # Record in buffer for statistics
            if is_terminated:
                num_goals_buffer.append(0)
                min_dist_buffer.append(min_dist)
                print ("Episode Finish ", policy)

            # Save trajactory sample <s, a, r> to the memory
            agent.append_sample(s_t, action_idx, r_t)

            # Update the cache
            t += 1

            if (is_terminated and t > agent.observe):
                # Every episode, agent learns from sample returns
                loss = agent.train_model()
                print(loss)

            # Save model every 10000 iterations
            if t % 10000 == 0:
                print("Save model")
                agent.save_model("models/a2c_beeline")

            state = ""
            if t <= agent.observe:
                state = "Observe mode"
            else:
                state = "Train mode"

            if (is_terminated):
                # Print performance statistics at every episode end
                print("TIME", t, "/ GAME", GAME, "/ STATE", state, "/ ACTION", action_idx, "/ REWARD", r_t, "/ MIN_DIST", min_dist, "/ STEPS", steps_taken, "/ LOSS", loss)

                # Save Agent's Performance Statistics
                if GAME % agent.stats_window_size == 0 and t > agent.observe:
                    print("Update Rolling Statistics")
                    agent.mavg_num_goals.append(np.mean(np.array(num_goals_buffer)))
                    agent.mavg_min_dist.append(np.mean(np.array(min_dist_buffer)))

                    # Reset rolling stats buffer
                    num_steps_buffer, min_dist_buffer = [], []

                    # Write Rolling Statistics to file
                    with open("statistics/a2c_goal_stats.txt", "w") as stats_file:
                        stats_file.write('mavg_num_goals: ' + str(agent.mavg_num_goals) + '\n')
                        stats_file.write('mavg_min_dist: ' + str(agent.mavg_min_dist) + '\n')

        # Episode Finish. Increment game count
        GAME += 1
