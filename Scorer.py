import numpy as np

from enum import Enum
from Random import RandomSelection
from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions


def counting_pieces(pre, curr):
    values = np.array([4, 1, 3, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0, 3])
    pieces_pre = pre[:, :, :12]
    pieces_curr = curr[:, :, :12]
    pieces_num_pre = np.array(
        [int(pieces_pre[:, :, i].sum()) for i in range(12)])
    pieces_num_curr = np.array(
        [int(pieces_curr[:, :, i].sum()) for i in range(12)])
    pieces_diff = pieces_num_curr - pieces_num_pre
    score = (pieces_diff * values).sum()
    return score


def step_score(pre, curr, player_id):
    if pre == None:
        return 0
    pre = pre[player_id]["partial_observation"]
    curr = curr[player_id]["partial_observation"]
    print(counting_pieces(pre, curr))
