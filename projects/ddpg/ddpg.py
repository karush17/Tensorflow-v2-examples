import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam

import gym
import numpy as np
import random
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--train_start', type=int, default=2000)

args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class Actor(tf.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.layer1 = Dense(32, activation='relu')
        self.layer2 = Dense(32, activation='relu')
        self.layer3 = Dense(self.action_dim, activation='tanh')
        self.custlambda = Lambda(lambda x: x*self.action_bound)
    
    def __call__(self, state):
        out = self.layer1(state)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.custlambda(out)
        return out
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.__call__(state)[0]

class Critic(tf.Module):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.s1 = Dense(32, activation='relu')
        self.s2 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        # self.c1 = concatenate([self.s2, self.a1], axis=-1)
        self.c2 = Dense(16, activation='relu')
        self.final = Dense(1, activation='linear')
    
    def __call__(self, states, actions):
        state_out = self.s1(states)
        state_out = self.s2(state_out)
        action_out = self.a1(actions)
        out = tf.concat([state_out, action_out], axis=-1)
        out = self.c2(out)
        out = self.final(out)
        return out

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.buffer = ReplayBuffer()
        self.actor_opt = Adam(args.actor_lr)
        self.critic_opt = Adam(args.critic_lr)
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.target_critic = Critic(self.state_dim, self.action_dim)

        self.target_update()

    def target_update(self):
        actor_vars = self.actor.trainable_variables
        target_actor_vars = self.target_actor.trainable_variables
        critic_vars = self.critic.trainable_variables
        target_critic_vars = self.target_critic.trainable_variables

        for var, target_var in zip(actor_vars, target_actor_vars):
            target_var.assign(args.tau*var + (1-args.tau)*target_var, True)

        for var, target_var in zip(critic_vars, target_critic_vars):
            target_var.assign(args.tau*var + (1-args.tau)*target_var, True)

    def td_target(self, rewards, q_values, dones):
        targets = np.copy(np.asarray(q_values))
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = args.gamma*q_values[i]
        return targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            target_q_values = self.target_critic(next_states, self.target_actor(next_states))
            td_targets = self.td_target(rewards, target_q_values, dones)

            with tf.GradientTape() as tape:
                v_pred = self.critic(states, actions)
                critic_loss = self.critic.compute_loss(v_pred, tf.stop_gradient(td_targets))
                grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
            
            s_actions = tf.convert_to_tensor(self.actor(states))
            with tf.GradientTape() as act_tape, tf.GradientTape() as crit_tape:
                crit_tape.watch(s_actions)
                q_vals = self.critic(states, s_actions)
                q_vals = tf.squeeze(q_vals)
                q_grads = crit_tape.gradient(q_vals, s_actions)
                grads = act_tape.gradient(self.actor(states), self.actor.trainable_variables, -q_grads)
                self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))
            
            self.target_update()

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            episode_reward, done = 0, False

            state = self.env.reset()
            bg_noise = np.zeros(self.action_dim)
            while not done:
                # self.env.render()
                action = self.actor.get_action(state)
                noise = self.ou_noise(bg_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, (reward+8)/8, next_state, done)
                bg_noise = noise
                episode_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size and self.buffer.size() >= args.train_start:
                self.replay()                
            print('EP{} EpisodeReward={}'.format(ep, episode_reward))


def main():
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()

