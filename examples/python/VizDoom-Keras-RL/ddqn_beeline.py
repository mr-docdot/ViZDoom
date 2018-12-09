#!/usr/bin/env python
from __future__ import print_function

import beeline
import random
import numpy as np
import tensorflow as tf
import vizdoom as vzd

from collections import deque
from keras import backend as K
from logger import Logger
from networks import Networks
from os import listdir
from os.path import isfile, join
from vizdoom import DoomGame, ScreenResolution


def get_sorted_wad_ids(wad_dir):
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_ids = [int(f.split('_')[1]) for f in wad_files]
    wad_ids.sort()

    return wad_ids


def setup_random_game():
    # Set up VizDoom Game
    game = DoomGame()
    game.load_config("./beeline.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)

    # Load generated map from WAD
    wad_dir = '../train_sptm/data/maps/out/'
    wad_ids = get_sorted_wad_ids(wad_dir)
    wad_id = random.choice(wad_ids)
    wad_path = '../train_sptm/data/maps/out/gen_{}_size_regular_mons_none_steepness_none.wad'.format(wad_id) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    game.new_episode()

    return game


class DoubleDQNAgent:

    def __init__(self, state_size, action_size, log_dir):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 100  # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = 50000  # number of previous transitions to remember

        # create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.logger = Logger(log_dir)
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_reward = []  # Moving Average of Kill Counts

        # Misc
        self.min_dist = -1

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)

        action = [int(x) for x in list('{0:03b}'.format(action_idx))]
        return action_idx, action

    def shape_reward(self, r_t, misc, game, cur_goal, is_new_goal, t):
        is_goal_reached = False

        cur_xy = np.array([misc[0], misc[1]])
        cur_dist = np.linalg.norm(cur_xy - cur_goal)
        damage_taken = game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)

        if damage_taken > 0:
            print('DAMAGE TAKEN')
            r_t = r_t - 30
            return r_t, cur_dist, False, True

        if is_new_goal:
            self.min_dist = cur_dist
        else:
            if cur_dist < self.min_dist:
                self.min_dist = cur_dist
                r_t = r_t + 0.01

            if cur_dist < 30:
                is_goal_reached = True
                r_t = r_t + 30
                print('GOAL REACHED')

        return r_t, cur_dist, is_goal_reached, False

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size)) 
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input) 
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        loss = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Set up game
    game = setup_random_game()
    game_state = game.get_state()
    misc = game_state.game_variables  # [cur_x, cur_y, angle]
    prev_misc = misc

    # Configure DQN networks
    action_size = 8
    state_size = (4, 3)

    log_dir = './logs/ddqn_beeline'
    agent = DoubleDQNAgent(state_size, action_size, log_dir)
    agent.model = Networks.dqn_novis(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.dqn_novis(state_size, action_size, agent.learning_rate)

    # Compute initial goal
    explored_goals = {}
    pick_new_goal = False
    cur_goal, rel_goal = beeline.compute_goal(game_state, None,
                                              explored_goals, True)

    # Compute initial state
    x_t = np.array(game_state.game_variables)  # [cur_x, cur_y, angle]
    x_t[:2] = rel_goal  # [rel_goal_x, rel_goal_y, angle]
    s_t = np.stack(([x_t]*4), axis=0)  # 4x3
    s_t = np.expand_dims(s_t, axis=0)  # 1x4x3

    cur_xy = np.array([misc[0], misc[1]])
    agent.min_dist = np.linalg.norm(cur_xy - cur_goal)

    is_terminated = game.is_episode_finished()
    should_terminate = False

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    episode_rewards = []
    last_total_reward = -1
    num_goals_reached = 0
    episode_length = 0

    while not game.is_episode_finished():
        loss = 0
        Q_max = 0
        r_t = 0

        # Compute new goal using beeline policy if necessary
        cur_goal, rel_goal = beeline.compute_goal(game_state, cur_goal,
                                                  explored_goals, pick_new_goal)

        # Epsilon Greedy
        action_idx, a_t = agent.get_action(s_t)

        game.set_action(a_t)
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        if cur_goal is None:
            is_terminated = True
            print('EARLY TERMINATION')
        else:
            is_terminated = game.is_episode_finished() or should_terminate
        should_terminate = False

        r_t = game.get_last_reward()  # each frame we get reward of 0.1, so 4 frames will be 0.4

        if (is_terminated):
            GAME += 1

            # Compute average reward of last episode
            last_total_reward = np.array(episode_rewards).sum()
            episode_rewards = []

            # Load random map
            game = setup_random_game()
            game_state = game.get_state()
            explored_goals = {}
            pick_new_goal = False
            cur_goal, rel_goal = beeline.compute_goal(game_state, None,
                                                      explored_goals, True)

        x_t1 = np.array(game_state.game_variables)  # [x, y, angle]
        x_t1[:2] = rel_goal  # [rel_goal_x, rel_goal_y, angle]
        x_t1 = x_t1.reshape(1, 1, 3)
        s_t1 = np.append(x_t1, s_t[:, :3, :], axis=1)
        misc = game_state.game_variables

        # Reward Shaping
        r_t, dist, is_goal_reached, should_terminate = agent.shape_reward(r_t, misc, game,
                                                                          cur_goal, pick_new_goal,
                                                                          t)
        pick_new_goal = is_goal_reached
        if is_goal_reached:
            num_goals_reached += 1

        episode_rewards.append(r_t)

        # Update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()

        s_t = s_t1
        t += 1
        episode_length += 1 * agent.frame_per_action

        # Save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/ddqn.h5", overwrite=True)

        # Print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            is_terminated = False

            agent.logger.log_scalar('Episode Length', episode_length, GAME)
            agent.logger.log_scalar('Episode Reward', last_total_reward, GAME)
            agent.logger.log_scalar('Number of Goals Reached', num_goals_reached, GAME)
            num_goals_reached = 0
            episode_length = 0

            print("TIME", t, "/ GAME", GAME, "/ STATE", state,
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", last_total_reward,
                  "/ Q_MAX %e" % np.max(Q_max), "/ LOSS", loss)
