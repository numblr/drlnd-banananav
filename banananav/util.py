import sys
import numpy as np

import matplotlib
# matplotlib.use("MacOSX")
# matplotlib.use("Agg")
from matplotlib import pyplot as plt


def print_progress(count, step, loss, scores, total=76, bar_len = 60, status=''):
    if count == 0:
        return

    filled_len = int(round(bar_len * step / float(total)))

    percents = round(100.0 * step / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    window = scores[-100:] if len(scores) > 100 else scores
    mean = np.mean(window) if len(window) > 0 else 0.0
    min = np.min(window) if len(window) > 0 else 0.0
    max = np.max(window) if len(window) > 0 else 0.0

    sys.stdout.write("{}/{} [{}] loss: {:+.3E} / score: {:+.3f}({:+.0f}/{:+.0f}/{:+.0f})\r"\
            .format(count, step, bar, loss, mean, min, scores[-1], max))
    sys.stdout.flush()


def plot(data):
    plt.plot(range(len(data)), data, c='b')
    plt.plot(range(len(data)), [ _mean(data, i, 10) for i in range(len(data)) ], c='g')
    plt.plot(range(len(data)), [ _mean(data, i, 100) for i in range(len(data)) ], c='r')
    plt.show()


def _mean(data, i, window):
    return np.mean(data[i-window:i]) if i > window else 0.0
