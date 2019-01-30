from pprint import pprint

import numpy as np

from banananav.environment import BananaEnv, BananaAgent, TestEnv
from banananav.training import DeepQLearner, DoubleDeepQLearner
from banananav.replaymemory import ReplayMemory
from banananav.util import print_progress, plot


path = "results/model.parameters"


env=BananaEnv()
learner = DoubleDeepQLearner(env=env)


scores = ()
losses = ()

episode_cnt = 0
episode_step = 0
max_avg_score = 0.0
for cnt, data in enumerate(learner.train(1800)):
    episode_step += 1
    loss, score, terminal = data

    if terminal:
        scores += (score, )
        losses += (loss.item(), )
        episode_cnt += 1
        episode_step = 0

    if terminal and np.mean(scores[:-25]) > max_avg_score:
        learner.save(path)
        max_avg_score = np.mean(scores[:-25])

    print_progress(episode_cnt, episode_step, loss, scores)
    if terminal:
        print("")

plot(scores)
plot(losses)


episode = env.generate_episode(learner.get_agent())
for count, step_data in enumerate(episode):
    pprint("step: " + str(count))
    pprint("score: " + str(env.get_score()))
    pprint("--")
