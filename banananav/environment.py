import pkg_resources

import random
import numpy as np

from unityagents import UnityEnvironment

PLATFORM_PATHS = ['Banana.app', 'Banana_Linux', 'Banana_Windows_x86', 'Banana_Windows_x86_64']

class BananaEnv:
    """Banana collection environment.

    The environment accepts actions and provides states and rewards in response.
    """

    def __init__(self):
        for path in PLATFORM_PATHS:
            try:
                unity_resource = pkg_resources.resource_filename('banananav', 'resources/' + path)
                self._env = UnityEnvironment(file_name=unity_resource)
                break
            except:
                pass

        if not hasattr(self, '_env'):
            raise Exception("No unity environment found, setup the environment as described in the README.")

        # get the default brain
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        self._info = None
        self._score = None

    def generate_episode(self, agent, max_steps=None):
        """Create a generator for and episode driven by an actor.
        Args:
            actor: An actor that provides the next action for a given state.
            max_steps: Maximum number of steps to take in the episode. If None,
                the episode is generated until a terminal state is reached.

        Returns:
            A generator providing a tuple of the step count and step data, where
            the step data is a tuple containing the current state, current action,
            reward, the next state and a flag whether the next state is terminal.
        """
        count = -1
        state = self.reset()
        is_terminal = False

        while not is_terminal and (max_steps is None or count < max_steps):
            count += 1

            action = agent.act(state)
            reward, next_state, is_terminal = self.step(action)

            step_data = (state, action, reward, next_state, is_terminal)

            state = next_state

            yield count, step_data

    def reset(self, train_mode=False):
        """Reset and initiate a new episode in the environment.

        Args:
            train_mode: Indicate if the environment should be initiated in
                training mode or not.

        Returns:
            The initial state of the episode (np.array).
        """
        if self._info is not None and not self._info.local_done[0]:
            raise Exception("Env is active, call terminate first")

        self._info = self._env.reset(train_mode=train_mode)[self._brain_name]
        self._score = 0

        return self._info.vector_observations[0]

    def step(self, action):
        """Execute an action.

        Args:
            action: An int representing the actionself.

        Returns:
            A tuple containing the reward (float), the next state (np.array) and
            a boolean indicating if the next state is terminal or not.
        """
        if self._info is None:
            raise Exception("Env is not active, call reset first")

        self._info = self._env.step(action)[self._brain_name]
        next_state = self._info.vector_observations[0]
        reward = self._info.rewards[0]
        is_terminal = self._info.local_done[0]
        self._score += reward

        return reward, next_state, is_terminal

    def terminate(self):
        self._info = None
        self._score = None

    def close(self):
        self._env.close()
        self._info = None

    def get_score(self):
        """Return the cumulative reward of the current episode."""
        return self._score

    def get_action_size(self):
        return self._brain.vector_action_space_size

    def get_state_size(self):
        return self._brain.vector_observation_space_size


class BananaAgent:
    """Agent based on a Q-function."""

    def __init__(self, Q, action_size, epsilon=0.0):
        """Initialize the agent.

        Args:
            Q: Q-function that is callable with a state and returns a 1-dim
                array-like containing the q-values for each action.
            action_size: The number of available actions.
            epsilon: propability for the agent to choose the action uniformly
                from the available actions instead of based on the Q-function,
                defaults to 0.0.
        """
        self._Q = Q
        self._action_size = action_size
        self._epsilon = epsilon

    def act(self, state):
        """Select an action for the given state.

        Args:
            state: The state to choose the action for.
        Returns:
            An int representing the action.
        """
        if self._epsilon == 0.0 or random.uniform(0, 1) > self._epsilon:
            action = np.argmax(self._Q(state))
            try:
                return action.item()
            except:
                return action

        return random.randint(0, self._action_size)


if __name__ == '__main__':
    # Run as > PYTHONPATH=".." python environment.py
    from pprint import pprint

    env = BananaEnv()
    print("Environment specs:")
    pprint(env.get_action_size())
    pprint(env.get_state_size())

    print("Reset:")
    pprint(env.reset())
    print("Action 0:")
    pprint(env.step(0))
    print("Action 1:")
    pprint(env.step(1))
    print("Action 2:")
    pprint(env.step(2))
    print("Action 3:")
    pprint(env.step(3))

    env.terminate()

    print("\n\n\nRun episode:")
    rand_Q = lambda s: np.random.rand(4)
    agent = BananaAgent(rand_Q, 4)
    episode = env.generate_episode(agent, max_steps=5)
    for count, step_data in episode:
        print("\n\nCount:")
        pprint(count)
        print("Step data:")
        pprint(step_data)

    env.close()
