import os 
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

class ProbabilityDistribution(tf.Module):
    def __call__(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.Module):
    def __init__(self, num_actions):
        super(Model, self).__init__()
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()
    
    def __call__(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)
    
    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.pol_optimizer = ko.RMSprop(lr=0.0007)
        self.val_optimizer = ko.RMSprop(lr=0.0007)
    
    def train(self, env, batch_sz=32, updates=1000):
        actions = np.empty((batch_sz,), dtpye=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        ep_rew = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_size):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if done[step]:
                    ep_rews.append(0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews)-1, ep_rews[-2]))
            
            _, next_value = self.model.action_values(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            with tf.GradientTape() as pol_tape, tf.GradientTape() as val_tape:
                pred_actions, pred_vals = self.model(observations)
                pol_loss = _logits_loss(acts_and_advs, pred_actions)
                val_loss = _value_loss(returns, pred_vals)
                pol_grads = pol_tape.gardient(pol_loss, self.model.trainable_variables)
                val_grads = val_tape.gradient(val_loss, self.model.trainable_variables)
                pol_optimizer.apply_gradients(zip(pol_grads, self.model.trainable_variables))
                val_optimizer.apply_gradients(zip(val_grads, self.model.trainable_variables))
        return ep_rews


    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, aixs=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma']*returns[t+1]*(1-dones[t])
        returns = returns[:-1]
        advs = rerturns - values
        return returns, advs
    
    def _value_loss(self, returns, value):
        return self.params['value']*kls.mean_squared_error(returns, value)
    
    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs,2,axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy']*entropy_loss

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model)
    rewards_hist = agent.train(env)
    print("Training Finished!")



