import numpy as np
def calculate_exploration_prob(loss_history, act_explor_prob,threshold):
    mean = np.mean(loss_history)

    variance = 0
    for i in loss_history:
        variance += np.square(i-mean)

    if threshold >= variance:
        act_explor_prob += 0.01
    else:
        act_explor_prob -= 0.01

    if act_explor_prob < 0:
        return 0
    elif act_explor_prob > 1:
        return 1
    else:
        return act_explor_prob


