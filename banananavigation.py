import numpy as np
import torch
from collections import namedtuple

from banananav.qmodel import BananaQModel, SimpleBananaQModel
from banananav.environment import BananaEnv, BananaAgent, TestEnv
from banananav.training import DeepQLearner, DoubleDeepQLearner
from banananav.replaymemory import ReplayMemory
from banananav.util import print_progress, plot, start_plot, save_plot

Settings=namedtuple('Settings', 'label color model learner parameters scores_fig loss_fig scores losses')

DQN_SETTINGS = Settings(**{
    'label': 'DQN',
    'color': 'c',
    'model': BananaQModel,
    'learner': DeepQLearner,
    'parameters': 'results/dqn_model.parameters',
    'scores_fig': 'results/dqn_model.scores.png',
    'loss_fig': 'results/dqn_model.losses.png',
    'scores': 'results/dqn_model.scores.raw',
    'losses': 'results/dqn_model.losses.raw'
})

DDQN_SETTINGS = Settings(**{
    'label': 'DDQN',
    'color': 'r',
    'model': BananaQModel,
    'learner': DoubleDeepQLearner,
    'parameters': 'results/ddqn_model.parameters',
    'scores_fig': 'results/ddqn_model.scores.png',
    'loss_fig': 'results/ddqn_model.losses.png',
    'scores': 'results/ddqn_model.scores.raw',
    'losses': 'results/ddqn_model.losses.raw'
})

DQN_SIMPLE_SETTINGS = Settings(**{
    'label': 'DQN simple',
    'color': 'g',
    'model': SimpleBananaQModel,
    'learner': DeepQLearner,
    'parameters':  'results/dqn_simple_model.parameters',
    'scores_fig':'results/dqn_simple_model.scores.png',
    'loss_fig': 'results/dqn_simple_model.losses.png',
    'scores': 'results/dqn_simple_model.scores.raw',
    'losses': 'results/dqn_simple_model.losses.raw'
})

DDQN_SIMPLE_SETTINGS = Settings(**{
    'label': 'DDQN simple',
    'color': 'y',
    'model': SimpleBananaQModel,
    'learner': DoubleDeepQLearner,
    'parameters': 'results/ddqn_simple_model.parameters',
    'scores_fig': 'results/ddqn_simple_model.scores.png',
    'loss_fig': 'results/ddqn_simple_model.losses.png',
    'scores': 'results/ddqn_simple_model.scores.raw',
    'losses': 'results/ddqn_simple_model.losses.raw'
})

MULTI_SCORES_PATH='results/multi_model.scores.png'

SETTINGS = {
    'DQN': DQN_SETTINGS,
    'DDQN': DDQN_SETTINGS,
    'DQN_SIMPLE': DQN_SIMPLE_SETTINGS,
    'DDQN_SIMPLE': DDQN_SIMPLE_SETTINGS
}


def run_learner(learner, result_path, episodes=750, checkpoint_window=33):
    scores = ()
    losses = ()

    episode_cnt = 0
    episode_step = 0
    max_avg_score = 0.0
    for cnt, data in enumerate(learner.train(episodes)):
        episode_step += 1
        loss, score, terminal = data

        if terminal:
            scores += (score, )
            losses += (loss.item(), )
            episode_cnt += 1
            episode_step = 0

        if terminal and np.mean(scores[:-checkpoint_window]) > max_avg_score:
            learner.save(result_path)
            max_avg_score = np.mean(scores[:-checkpoint_window])

        print_progress(episode_cnt, episode_step, loss, scores)
        if terminal:
            print("")

    return scores, losses


def replay(env, settings):
    print("Replay from files")

    learner = settings.learner(env=env, model=settings.model)
    learner.load(settings.parameters)
    replay_agent = learner.get_agent()
    episode = env.generate_episode(replay_agent)
    for count, step_data in enumerate(episode):
        print("=== step: " + str(count))
        print("score:    " + str(env.get_score()))

def learn(env, settings, episodes):
    print("\nStart learning with " + settings.label + "\n")

    learner = settings.learner(env=env, model=settings.model)
    scores, losses = run_learner(learner, settings.parameters, episodes=episodes)

    print("\nStore results for " + settings.label + "\n")

    plot(scores, path=settings.scores_fig)
    plot(losses, path=settings.loss_fig)

    torch.save(scores, settings.scores)
    torch.save(losses, settings.losses)

    return scores, losses



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--replay',
            help='Only replay from the stored parameters',
            action='store_true', required=False)
    parser.add_argument('-s','--settings',
            help='Comma separated string with settings from [DQN, DDQN, DQN_SIMPLE, DDQN_SIMPLE]',
            default='DQN,DDQN,DQN_SIMPLE,DDQN_SIMPLE', required=False)
    parser.add_argument('-n','--episodes',
            help='Number of episodes used for training',
            type=int, default=2500, required=False)

    args = parser.parse_args()
    selected_settings = [ SETTINGS[s.strip()] for s in args.settings.split(",")]

    env=BananaEnv()

    if args.replay:
        for settings in selected_settings:
            replay(env, settings)
    else:
        results = [ learn(env, settings, args.episodes) + (settings.label, settings.color)
                for settings in selected_settings ]

        start_plot()
        for scores, losses, label, color in results:
            plot(scores, windows=[100], colors=[color], labels=[label], path=None)
        save_plot(MULTI_SCORES_PATH, loc='lower right')
