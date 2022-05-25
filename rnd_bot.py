import numpy as np

from stratego_env import ObservationComponents


def RandomSelection(current_player, obs_from_env):
    valid_actions_mask = obs_from_env[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]

    valid_actions_mask_1D = np.reshape(valid_actions_mask, -1)
    actions = []
    for i in range(valid_actions_mask_1D.shape[0]):
        if valid_actions_mask_1D[i] == 1:
            actions += [i]

    chosen_action_index = np.random.choice(actions)
    return chosen_action_index
