from pprint import pprint
import numpy as np
from collections import deque

from banananav.environment import BananaEnv, BananaAgent, TestEnv
from banananav.training import DeepQLearner
from banananav.replaymemory import ReplayMemory

env = TestEnv()
# learner = DeepQLearner(env=env)
learner = DeepQLearner(env=env)
scores_window = deque(maxlen=100)
for cnt, data in enumerate(learner.train(2000)):
    loss, score, terminal = data
    if terminal:
        scores_window.append(score)
        pprint("{} - loss: {:+.3f} / score: {:+.3f}({:+.3f})".format(cnt, loss, np.mean(scores_window), score))

episode = env.generate_episode(learner.get_agent())
for count, step_data in enumerate(episode):
    pprint("step: " + str(count))
    pprint("score: " + str(env.get_score()))
    pprint("--")
