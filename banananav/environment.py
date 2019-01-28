import pkg_resources
import random
import torch
import numpy as np

from unityagents import UnityEnvironment
from unityagents.exception import UnityEnvironmentException

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
                print("Environment loaded from " + path)
                break
            except UnityEnvironmentException as e:
                print("Attempted to load " + path + ":")
                print(e)
                print("")
                pass

        if not hasattr(self, '_env'):
            raise Exception("No unity environment found, setup the environment as described in the README.")

        # get the default brain
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        self._info = None
        self._score = None

    def generate_episode(self, agent, max_steps=None, train_mode=False):
        """Create a generator for and episode driven by an actor.
        Args:
            actor: An actor that provides the next action for a given state.
            max_steps: Maximum number of steps (int) to take in the episode. If
                None, the episode is generated until a terminal state is reached.

        Returns:
            A generator providing a tuple of the current state, the action taken,
            the obtained reward, the next state and a flag whether the next
            state is terminal or not.
        """
        state = self.reset(train_mode=train_mode)
        is_terminal = False
        count = 0

        while not is_terminal and (max_steps is None or count < max_steps):
            action = agent.act(state)
            reward, next_state, is_terminal = self.step(action)

            step_data = (state, action, reward, next_state, is_terminal)

            state = next_state
            count += 1

            yield step_data

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
        if not torch.is_tensor(state):
            try:
                state = torch.from_numpy(state)
            except:
                state = torch.from_numpy(np.array(state, dtype=np.float))

        state = state.float()

        if self._epsilon == 0.0 or random.uniform(0, 1) > self._epsilon:
            with torch.no_grad():
                return torch.argmax(self._Q(state)).item()

        return np.random.randint(self._action_size)


import gym
# import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
# print(matplotlib.rcsetup.non_interactive_bk)
# print(matplotlib.rcsetup.all_backends)
# matplotlib.use("MacOSX")
# matplotlib.use("Agg")
# from matplotlib import pyplot as plt
# from pyvirtualdisplay import Display

class TestEnv():
    """Environment with known performance for testing purposes."""

    def __init__(self):
        # self.display = Display(visible=0, size=(1400, 900))
        # self.display.start()

        # is_ipython = 'inline' in plt.get_backend()
        # if is_ipython:
        #     from IPython import display
        #
        # plt.ion()

        self.env = gym.make('LunarLander-v2')
        self.env.seed(0)

        self._score = 0

    def generate_episode(self, agent, max_steps=None, train_mode=False):
        state = self.env.reset()
        self._score = 0

        # img = plt.imshow(self.env.render(mode='rgb_array'))

        cnt = 0
        done = False
        while not done and cnt < 1000:
            cnt += 1
            action = agent.act(state)

            # img.set_data(self.env.render(mode='rgb_array'))
            # plt.axis('off')
            next_state, reward, done, _ = self.env.step(action)

            step_data = (state, action, reward, next_state, done)

            state = next_state
            self._score += reward

            yield step_data

    def get_state_size(self):
        return self.env.observation_space.shape[0]

    def get_action_size(self):
        return self.env.action_space.n

    def get_score(self):
        return self._score

    def close(self):
        self.env.close()


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
    rand_Q = lambda s: torch.rand(4)
    agent = BananaAgent(rand_Q, 4)
    episode = enumerate(env.generate_episode(agent))
    for count, step_data in episode:
        print("\n\nCount:")
        pprint(count)
        print("Step data:")
        pprint(step_data)

    env.close()
