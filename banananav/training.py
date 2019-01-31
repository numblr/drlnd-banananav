import torch
from torch import optim
import torch.nn.functional as F

from banananav.qmodel import BananaQModel
from banananav.environment import BananaEnv, BananaAgent
from banananav.replaymemory import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepQLearner():
    """Implementation of the DQN learning algorithm with experience replay.
    """

    def __init__(self, env=None, model=BananaQModel, memory=ReplayMemory(int(3e4)),
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=1e-4, decay=0.001,
            epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
            gamma=0.99, tau=1e-3):
        self._memory = memory
        # Don't instantiate as default as the constructor already starts the unity environment
        self._env = env if env is not None else BananaEnv()

        self._state_size = self._env.get_state_size()
        self._actions = self._env.get_action_size()

        self._batch_steps = batch_steps
        self._batch_size = batch_size
        self._batch_repeat = batch_repeat

        self._epsilon_start=epsilon_start
        self._epsilon_min=epsilon_min
        self._epsilon_decay=epsilon_decay
        self._gamma = gamma
        self._tau = tau

        self._qnetwork_local = model(self._state_size, self._actions).to(device)
        self._qnetwork_target = model(self._state_size, self._actions).to(device)
        self._optimizer = optim.Adam(self._qnetwork_local.parameters(), lr=lr,
            amsgrad=True)

        self._qnetwork_local.eval()
        self._qnetwork_target.eval()

    def save(self, path):
        """Store the learning result.

        Store the parameters of the current Q-function approximation to the given path.
        """
        torch.save(self._qnetwork_local.state_dict(), path)

    def load(self, path):
        """Load learning results.

        Load the parameters from the given path into the current and target
        Q-function approximator.
        """
        self._qnetwork_local.load_state_dict(torch.load(path))
        self._qnetwork_target.load_state_dict(torch.load(path))
        self._qnetwork_local.to(device)
        self._qnetwork_target.to(device)

    def get_agent(self, epsilon=0.0):
        """Return an agent based on the parameters of the current Q-function approximation.
        """
        return BananaAgent(self._qnetwork_local, self._env.get_action_size(),
                epsilon=epsilon)

    def train(self, num_episodes=100):
        episodes = ( self._env.generate_episode(
                        self.get_agent(self._get_epsilon(cnt)), train_mode=True)
                for cnt in range(num_episodes) )
        steps = ( (cnt, step_cnt, step_data)
                for cnt, episode in enumerate(episodes)
                for step_cnt, step_data in enumerate(episode) )

        for episode, step, step_data in steps:
            self._memory.add(*step_data)

            if (step % self._batch_steps == 0 or self._is_terminal(step_data)) \
                    and self._memory.size() >= self._batch_size:
                for i in range(1 if self._memory.size() < 1000 else self._batch_repeat):
                    loss = self._train_from_memory()
                    self._update_target()
                yield loss, self._env.get_score(), self._is_terminal(step_data)

    def _get_epsilon(self, cnt):
        return max(self._epsilon_min, self._epsilon_decay ** cnt * self._epsilon_start)

    def _train_from_memory(self):
        self._qnetwork_local.train()

        batch = self._memory.sample(self._batch_size)
        loss = self._calculate_loss(*zip(*batch))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._qnetwork_local.eval()

        return loss

    def _calculate_loss(self, states, actions, rewards, next_states, is_terminal):
        states, next_states, rewards, is_terminal = self._to_tensor(states, next_states, rewards, is_terminal)
        actions = self._to_tensor(actions, dtype=torch.long)[0]

        rewards = rewards.unsqueeze(1)
        is_terminal = is_terminal.unsqueeze(1)
        actions = actions.unsqueeze(1)

        Q_target_next = self._qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (self._gamma * Q_target_next * (1 - is_terminal))
        Q_predicted = self._qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_target, Q_predicted)

        # Validate dimensions
        assert Q_predicted.size()[0] == states.size()[0]
        assert Q_predicted.size()[1] == 1
        assert Q_predicted.size() == Q_target_next.size() == Q_target.size() \
                == rewards.size() == is_terminal.size() == actions.size()

        return loss

    def _update_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        parameters = zip(self._qnetwork_target.parameters(), self._qnetwork_local.parameters())

        for target_param, local_param in parameters:
            update = self._tau * local_param.data + (1.0 - self._tau) * target_param.data
            target_param.data.copy_(update)

    def _is_terminal(self, state_data):
        return state_data[-1]

    def _get_reward(self, state_data):
        return state_data[2]

    def _to_tensor(self, *arrays, dtype=torch.float):
        return tuple(torch.tensor(a).to(device, dtype=dtype) for a in arrays)


class DoubleDeepQLearner(DeepQLearner):
    """Implementation of the Double-DQN learning algorithm with experience replay.
    """

    def __init__(self, env=None, model=BananaQModel, memory=ReplayMemory(int(3e4)),
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=1e-4, decay=0.001,
            epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
            gamma=0.99, tau=1e-3):
        super(DoubleDeepQLearner, self).__init__(env=env, model=model, memory=memory,
                batch_steps=batch_steps, batch_size=batch_size, batch_repeat=batch_repeat,
                lr=lr, decay=decay,
                epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                gamma=gamma, tau=tau)

    def _calculate_loss(self, states, actions, rewards, next_states, is_terminal):
        states, next_states, rewards, is_terminal = self._to_tensor(states, next_states, rewards, is_terminal)
        actions = self._to_tensor(actions, dtype=torch.long)[0]

        rewards = rewards.unsqueeze(1)
        is_terminal = is_terminal.unsqueeze(1)
        actions = actions.unsqueeze(1)

        Q_local_next_choices = self._qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_target_next = self._qnetwork_target(next_states).detach().gather(1, Q_local_next_choices)
        Q_target = rewards + (self._gamma * Q_target_next * (1 - is_terminal))
        Q_predicted = self._qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_target, Q_predicted)

        # Validate dimensions
        assert Q_local_next_choices.size() == actions.size()
        assert Q_predicted.size()[0] == states.size()[0]
        assert Q_predicted.size()[1] == 1
        assert Q_predicted.size() == Q_target_next.size() == Q_target.size() \
                == rewards.size() == is_terminal.size() == actions.size()

        return loss
