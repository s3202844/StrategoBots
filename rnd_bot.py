import numpy as np


def random_act(obs_from_env):
    current_player = list(obs_from_env.keys())[0]
    valid_actions_mask = obs_from_env[current_player]["valid_actions_mask"]

    valid_actions_mask_1D = np.reshape(valid_actions_mask, -1)
    actions = []
    for i in range(valid_actions_mask_1D.shape[0]):
        if valid_actions_mask_1D[i] == 1:
            actions += [i]

    chosen_action_index = np.random.choice(actions)
    return chosen_action_index
