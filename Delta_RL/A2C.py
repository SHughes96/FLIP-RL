import jax
import jax.numpy as jnp
import optax
import numpy as np
import gym
from flax import linen as nn
from typing import Sequence

# Define the actor and critic networks
class Actor(nn.Module):
    hidden_sizes: Sequence[int]
    action_dim: int

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        self.output_layer = nn.Dense(self.action_dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        logits = self.output_layer(x)
        return logits  # Unnormalized logits for the action probabilities

class Critic(nn.Module):
    hidden_sizes: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        self.output_layer = nn.Dense(1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        value = self.output_layer(x)
        return value  # Scalar value estimate


# A2C class to encapsulate the training logic
class A2C:
    def __init__(self, env_name, actor_hidden_sizes, critic_hidden_sizes, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.gamma = gamma

        # Initialize actor and critic networks
        self.actor = Actor(actor_hidden_sizes, self.action_dim)
        self.critic = Critic(critic_hidden_sizes)

        self.rng = jax.random.PRNGKey(0)
        sample_input = jnp.ones((1, self.obs_dim))

        self.actor_params = self.actor.init(self.rng, sample_input)
        self.critic_params = self.critic.init(self.rng, sample_input)

        # Optimizers
        self.actor_optimizer = optax.adam(actor_lr)
        self.critic_optimizer = optax.adam(critic_lr)

        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    def select_action(self, params, state):
        """Select an action and return the action and action probabilities."""
        state = state.reshape(1, -1)  # Ensure state has batch dimension
        logits = self.actor.apply(params, state)
        action_probs = jax.nn.softmax(logits)
        
        #self.rng, key = jax.random.split(self.rng)
        #action = jax.random.categorical(key, logits)
        action = jax.random.categorical(jax.random.PRNGKey(0), logits)
    
        # Convert action array to scalar
        action = int(action[0])
        
        return int(action), action_probs

    @jax.jit
    def actor_loss(self, actor_params, critic_params, states, actions, advantages):
        def policy_loss_fn(state, action, advantage):
            logits = self.actor.apply(actor_params, state)
            log_prob = jax.nn.log_softmax(logits)[action]
            return -log_prob * advantage

        loss = jax.vmap(policy_loss_fn)(states, actions, advantages).mean()
        return loss

    @jax.jit
    def critic_loss(self, critic_params, states, returns):
        def value_loss_fn(state, ret):
            value = self.critic.apply(critic_params, state)
            return jnp.square(value - ret)

        loss = jax.vmap(value_loss_fn)(states, returns).mean()
        return loss

    @jax.jit
    def update_actor(self, actor_params, actor_opt_state, critic_params, states, actions, advantages):
        loss, grads = jax.value_and_grad(self.actor_loss)(actor_params, critic_params, states, actions, advantages)
        updates, new_actor_opt_state = self.actor_optimizer.update(grads, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, updates)
        return loss, new_actor_params, new_actor_opt_state

    @jax.jit
    def update_critic(self, critic_params, critic_opt_state, states, returns):
        loss, grads = jax.value_and_grad(self.critic_loss)(critic_params, states, returns)
        updates, new_critic_opt_state = self.critic_optimizer.update(grads, critic_opt_state)
        new_critic_params = optax.apply_updates(critic_params, updates)
        return loss, new_critic_params, new_critic_opt_state

    def train(self, num_episodes=1000, batch_size=5):
        for episode in range(num_episodes):
            obs = self.env.reset()
            
            # Convert obs to numpy if it's not flat
            if not isinstance(obs, np.ndarray):
                obs = np.asarray(obs[0], dtype=np.float32)  # Ensure it's a float32 array
            print('converted obs', obs)
            obs = jnp.array(obs)  # Convert to JAX array
            done = False
            rewards = []
            actions = []
            states = []
            values = []

            # Collect batch of data
            for _ in range(batch_size):
                states.append(obs)
                action, action_probs = self.select_action(self.actor_params, obs)
                #obs, reward, done, _ = self.env.step(action)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                # Flatten and ensure the obs is valid
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)

                obs = jnp.array(obs)
                rewards.append(reward)
                actions.append(action)

                value = self.critic.apply(self.critic_params, obs.reshape(1, -1))  # Reshape obs for critic
                values.append(value)

                if done:
                    break
                
            # Convert states and actions to jnp.array
            states = jnp.array(states)
            actions = jnp.array(actions)

            # Compute returns and advantages
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)

            returns = jnp.array(returns)
            values = jnp.squeeze(jnp.array(values))
            advantages = returns - values
            
            # Ensure states, actions, and advantages are JAX arrays with proper dtype
            states = jnp.asarray(states, dtype=jnp.float32)
            actions = jnp.asarray(actions, dtype=jnp.int32)
            advantages = jnp.asarray(advantages, dtype=jnp.float32)

            # Update the actor and critic
            actor_loss, self.actor_params, self.actor_opt_state = self.update_actor(
                self.actor_params, self.actor_opt_state, self.critic_params, jnp.array(states), jnp.array(actions), advantages
            )

            critic_loss, self.critic_params, self.critic_opt_state = self.update_critic(
                self.critic_params, self.critic_opt_state, jnp.array(states), returns
            )

            print(f"Episode {episode}: Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

# Example usage
if __name__ == "__main__":
    env_name = "CartPole-v1"
    actor_hidden_sizes = [64, 64]
    critic_hidden_sizes = [64, 64]
    agent = A2C(env_name, actor_hidden_sizes, critic_hidden_sizes)
    agent.train(num_episodes=500, batch_size=5)


