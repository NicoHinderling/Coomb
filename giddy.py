import math
import random
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

random.seed(3)

# class Bandit(object):
#     def __init__(self, num_options=2, prior=(1.0, 1.0)):
#         self.trials = np.zeros(shape=(num_options,), dtype=int)
#         self.successes = np.zeros(shape=(num_options,), dtype=int)
#         self.num_options = num_options
#         self.prior = prior

#     def update(self, choice, conv):
#         self.trials[choice] = self.trials[choice] + 1
#         if (conv):
#             self.successes[choice] = self.successes[choice] + 1

#     def evaluate(self):
#         sampled_theta = []
#         for i in range(self.num_options):
#             dist = beta(self.prior[0] + self.successes[i],
#                         self.prior[1] + self.trials[i] - self.successes[i])
#             sampled_theta += [dist.rvs()]
#         return sampled_theta.index(max(sampled_theta))

# #

# theta = (0.3, 0.4)

# def click(c):
#     if random.random() < theta[c]:
#         return True
#     else:
#         return False

# #

# N = 10000
# incr_trials = np.zeros(shape=(N, 2))
# successes = np.zeros(shape=(N, 2))

# b = Bandit()
# for i in range(N):
#     choice = b.evaluate()
#     conv = click(choice)
#     b.update(choice, conv)
#     incr_trials[i] = b.trials
#     successes[i] = b.successes

# #

# n = np.arange(N) + 1
# f, ax = plt.subplots(figsize=(15, 7))
# ax.set(xscale='log', yscale='log')
# ax.set_title('Pulls Per Arm with Respect to Trial Number')
# ax.set_xlabel('Trials')
# ax.set_ylabel('Pulls')
# plt.plot(n, incr_trials[:, 0], label='Theta 0')
# plt.plot(n , incr_trials[:, 1], label='Theta 1')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# #

symbol_list = ['btc', 'banx', 'clam', 'pro']
sym_dict = {}
sym_key_val = []
for i, sym in enumerate(symbol_list):
    sym_dict[sym] = i
    sym_key_val.append((sym, i))
    
df = {'open_price': {'btc': [263.96, 263.767, 263.676, 263.612, 263.66, 263.929, 263.759, 263.68, 263.594, 263.103, 263.003, 263.274, 263.508, 263.275, 262.916, 261.476, 261.846, 261.989, 261.807, 261.296, 261.495, 261.866, 261.569, 261.493, 261.224, 260.988, 261.373, 261.457, 261.062, 261.476, 261.477, 261.417, 261.354, 261.479, 261.685, 261.797, 262.042, 262.148, 262.186, 262.2, 262.261, 262.44, 262.42, 262.294, 261.848, 261.829, 261.774, 261.515, 261.494, 261.444, 261.457, 261.069, 261.035, 261.015, 260.968, 261.172, 261.152, 261.215, 261.304, 261.275, 261.346, 261.272, 261.417, 261.306, 261.325, 258.075, 257.59],
                     'banx': [2.00876, 2.00463, 2.00394, 2.00368, 2.00382, 2.00586, 2.00457, 2.00397, 2.00331, 1.99958, 1.99944, 2.00088, 2.00266, 2.00089, 1.99816, 1.98722, 1.99003, 1.99112, 1.98973, 1.98585, 1.98736, 1.99018, 1.98792, 1.98735, 1.9853, 1.98351, 1.98643, 1.98707, 1.98407, 1.98722, 1.98723, 1.98677, 1.98629, 1.98724, 1.98881, 1.98966, 1.99152, 1.99232, 1.99261, 1.99272, 1.99318, 1.99454, 1.99439, 1.99343, 1.99004, 1.9899, 1.98948, 1.98751, 1.98735, 1.98697, 1.98707, 1.98412, 1.98387, 1.98371, 1.98336, 1.98491, 1.98476, 1.98523, 1.98591, 1.98569, 1.98623, 1.98567, 1.98677, 1.98593, 1.98607, 1.96137, 1.96137], 
                     'clam': [3.68495, 3.66689, 3.64847, 3.648, 3.64825, 3.65199, 3.59006, 3.64865, 3.61372, 3.51119, 3.62725, 3.66014, 3.43464, 3.43143, 3.65322, 3.63321, 3.22941, 3.23695, 3.22346, 3.18727, 3.30382, 3.19476, 3.19108, 3.39626, 3.2421, 3.23912, 3.34451, 3.24486, 3.23651, 3.34084, 3.34085, 3.34068, 3.5142, 3.51588, 3.524, 3.52551, 3.52879, 3.53021, 3.55061, 3.53093, 3.53175, 3.48246, 3.48219, 3.55206, 3.54602, 3.47671, 3.47267, 3.54104, 3.5408, 3.46954, 3.46972, 3.46457, 3.45945, 3.4558, 3.45276, 3.45591, 3.45564, 3.51962, 3.52083, 3.52044, 3.5103, 3.50878, 3.51038, 3.43461, 3.43825, 3.3785, 3.46116],
                     'pro': [2.0153, 2.01431, 2.01482, 2.01368, 2.01359, 2.01433, 2.01311, 2.01336, 2.01302, 2.01234, 2.01224, 2.01228, 2.0119, 2.0113, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156]
                    },
      'close_price': {'btc': [263.96, 263.779, 263.569, 263.66, 263.794, 263.816, 263.86, 263.778, 263.271, 263.003, 263.209, 263.434, 263.37, 263.308, 262.347, 261.952, 261.902, 261.901, 261.399, 261.409, 261.551, 261.813, 261.421, 261.62, 260.905, 261.251, 261.501, 261.472, 261.123, 261.6, 261.51, 261.349, 261.384, 261.536, 261.664, 261.955, 262.18, 262.201, 262.212, 262.241, 262.418, 262.363, 262.287, 262.18, 261.744, 261.798, 261.51, 261.524, 261.497, 261.406, 261.06, 261.01, 261.023, 261.048, 260.988, 261.23, 261.223, 261.192, 261.302, 261.35, 261.298, 261.411, 261.32, 261.299, 259.499, 257.784, 257.533],
                      'banx': [2.0061, 2.00472, 2.00312, 2.00368, 2.00483, 2.005, 2.00534, 2.00404, 2.00086, 1.99958, 2.00039, 2.0021, 2.00161, 2.00114, 1.99384, 1.99084, 1.99046, 1.99045, 1.98663, 1.98671, 1.98779, 1.98978, 1.9868, 1.98831, 1.98288, 1.98551, 1.98741, 1.98719, 1.98453, 1.98816, 1.98748, 1.98625, 1.98652, 1.98767, 1.98865, 1.99086, 1.99257, 1.99273, 1.99281, 1.99303, 1.99438, 1.99396, 1.99338, 1.99284, 1.98925, 1.98966, 1.98748, 1.98758, 1.98738, 1.98669, 1.98406, 1.98368, 1.98377, 1.98469, 1.98351, 1.98535, 1.98529, 1.98506, 1.9859, 1.98626, 1.98586, 1.98672, 1.98603, 1.98587, 1.97219, 1.95916, 1.95725],
                      'clam': [3.56719, 3.64989, 3.64699, 3.648, 3.65011, 3.66758, 3.65114, 3.61502, 3.51343, 3.51119, 3.65436, 3.66224, 3.66069, 3.43186, 3.64532, 3.25679, 3.23574, 3.22461, 3.18856, 3.30299, 3.30446, 3.19411, 3.39533, 3.24699, 3.23815, 3.2423, 3.34615, 3.24505, 3.23727, 3.34242, 3.34127, 3.51413, 3.5146, 3.51666, 3.52372, 3.52762, 3.53065, 3.53095, 3.55096, 3.53149, 3.55375, 3.48144, 3.55197, 3.551, 3.54459, 3.47299, 3.46917, 3.54116, 3.47025, 3.46904, 3.46445, 3.46091, 3.45654, 3.455, 3.45326, 3.45667, 3.51973, 3.51931, 3.52081, 3.52144, 3.5093, 3.43548, 3.43819, 3.43791, 3.39714, 3.46375, 3.44981],
                      'pro': [2.01461, 2.01459, 2.0132, 2.01368, 2.01424, 2.01407, 2.01272, 2.01346, 2.0125, 2.01234, 2.0123, 2.01215, 2.01146, 2.01042, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156, 2.01156]
                    }
     }
def run(df):
    random.shuffle(sym_key_val)
    #trading_days = len(df['open_price'].ix[:,0])
    trading_days = len(df['open_price'])
    trading_days
#     num_stocks = len(df['open_price'].columns)
    num_stocks = len(df['open_price'])

    reward = lambda choice, t: payoff(df, t, sym_key_val[choice][0])
    action_reward = lambda j: sum([reward(j,t) for t in range(trading_days)])

    best_action = max(range(num_stocks), key=action_reward)
    best_action_reward_cum = action_reward(best_action)

    cum_reward = 0
    t = 0
    ucb1gen = ucb1(num_stocks, reward)
    for (action, reward, ucbs) in ucb1gen:
        cum_reward += reward
        t += 1
        if t == trading_days:
            break
            
    return cum_reward, best_action_reward_cum, ucbs, sym_key_val[best_action][0]

def upper_bound(step, num_tests):
    return math.sqrt(2.0 * math.log(step + 1) / num_tests)

def ucb1(num_stocks, reward):
    payoff_sums = [0.0] * num_stocks
    num_tests = [1] * num_stocks
    ucbs = [0.0] * num_stocks

    for t in range(num_stocks):
        payoff_sums[t] = reward(t,t)
        yield t, payoff_sums[t], ucbs

    t = num_stocks

    while True:
        ucbs = [payoff_sums[i] / num_tests[i] + upper_bound(t, num_tests[i]) for i in range(num_stocks)]
        action = max(range(num_stocks), key=lambda i: ucbs[i])
        reward_val = reward(action, t)

        for a in range(num_stocks):
            num_tests[a] += 1
            payoff_sums[a] += reward(a, t)

        yield action, reward_val, ucbs
        t = t + 1

def payoff(df, t, stock, cash_am=1.0):
#     open_p, close_p = df['open_price'].ix[:,sym_dict[stock]][t], \
#                            df['close_price'].ix[:,sym_dict[stock]][t]
    open_p, close_p = df['open_price'][stock][t], df['close_price'][stock][t]
    
    
    # allows for purchasing fraction of shares     
    shares_purchased = cash_am / open_p
    cash_from_sale = shares_purchased * close_p

    return cash_from_sale - cash_am


ucb_list = lambda L: ', '.join(['%.3f' % x for x in L])

def mean(lst):
    sm = 0
    count = 0
    for x in lst:
        sm += x
        count += 1
    return 0 if count == 0 else float(sm) / count

def stats(lst):
    vals = [x for x in lst]
    avg = mean(vals)
    devs = [(x-avg)*(x-avg) for x in vals]
    return (avg, mean(devs))

reward, best_action_reward, ucbs, best_stock = run(df)

mins = len(df['open_price']['btc'])

reward = lambda choice, t: payoff(df, t, choice)
action_rewards = lambda s: np.array([reward(s,t) for t in range(mins)])
xs = np.array(list(range(mins)))

f, ax = plt.subplots(figsize=(15, 7))

ax.set_title('Rewards Over Time')
ax.set_xlabel('5 Minute Intervals')
ax.set_ylabel('Reward')
ax.set_xlim(0, mins-1)
for sym in symbol_list:
    plt.plot(xs, np.cumsum(action_rewards(sym)), label=sym)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


