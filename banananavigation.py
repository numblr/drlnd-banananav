from pprint import pprint

from banananav.environment import BananaEnv, BananaAgent, TestEnv
from banananav.training import DeepQLearner
from banananav.replaymemory import ReplayMemory
from banananav.util import print_progress, plot


learner = DeepQLearner()

scores = ()
losses = ()

episode_cnt = 0
episode_step = 0
for cnt, data in enumerate(learner.train(1500)):
    episode_step += 1
    loss, score, terminal = data

    if terminal:
        scores += (score, )
        losses += (loss.item(), )
        episode_cnt += 1
        episode_step = 0

    print_progress(episode_cnt, episode_step, loss, scores)
    if terminal:
        print("")

plot(scores)
plot(losses)


env = BananaEnv()
episode = env.generate_episode(learner.get_agent())
for count, step_data in enumerate(episode):
    pprint("step: " + str(count))
    pprint("score: " + str(env.get_score()))
    pprint("--")
