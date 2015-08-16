import math
import random
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

random.seed(3)

class Bandit(object):
    def __init__(self, num_options=2, prior=(1.0, 1.0)):
        self.trials = np.zeros(shape=(num_options,), dtype=int)
        self.successes = np.zeros(shape=(num_options,), dtype=int)
        self.num_options = num_options
        self.prior = prior

    def update(self, choice, conv):
        self.trials[choice] = self.trials[choice] + 1
        if (conv):
            self.successes[choice] = self.successes[choice] + 1

    def evaluate(self):
        sampled_theta = []
        for i in range(self.num_options):
            dist = beta(self.prior[0] + self.successes[i],
                        self.prior[1] + self.trials[i] - self.successes[i])
            sampled_theta += [dist.rvs()]
        return sampled_theta.index(max(sampled_theta))

#

theta = (0.3, 0.4)

def click(c):
    if random.random() < theta[c]:
        return True
    else:
        return False

#

N = 10000
incr_trials = np.zeros(shape=(N, 2))
successes = np.zeros(shape=(N, 2))

b = Bandit()
for i in range(N):
    choice = b.evaluate()
    conv = click(choice)
    b.update(choice, conv)
    incr_trials[i] = b.trials
    successes[i] = b.successes

#

n = np.arange(N) + 1
f, ax = plt.subplots(figsize=(15, 7))
ax.set(xscale='log', yscale='log')
ax.set_title('Pulls Per Arm with Respect to Trial Number')
ax.set_xlabel('Trials')
ax.set_ylabel('Pulls')
plt.plot(n, incr_trials[:, 0], label='Theta 0')
plt.plot(n , incr_trials[:, 1], label='Theta 1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#





