from pprint import pprint
import numpy as np

from banananav.environment import BananaEnv, BananaAgent

env = BananaEnv()
randomAgent = BananaAgent(lambda s: np.random.rand(env.get_action_size()))

episode = env.generate_episode(randomAgent)
for count, step_data in episode:
    pprint("action: " + str(count))
    pprint("action: " + str(step_data[1]))
