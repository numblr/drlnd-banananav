from pprint import pprint
import numpy as np
from collections import deque

from banananav.environment import BananaEnv, BananaAgent, TestEnv
from banananav.training import DeepQLearner
from banananav.replaymemory import ReplayMemory

learner = DeepQLearner()
scores_window = deque(maxlen=100)
for cnt, data in enumerate(learner.train(200)):
    loss, score, terminal = data
    if terminal:
        scores_window.append(score)
        pprint("{} - loss: {:+.3f} / score: {:+.3f}({:+.3f})".format(cnt, loss, np.mean(scores_window), score))

env = BananaEnv()
episode = env.generate_episode(learner.get_agent())
for count, step_data in enumerate(episode):
    pprint("step: " + str(count))
    pprint("score: " + str(env.get_score()))
    pprint("--")
