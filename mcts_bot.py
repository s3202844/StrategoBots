import math
import copy
import random
import numpy as np

from scorer import step_score
from rnd_bot import random_act


# hyperparameters
global depth, epsilon, min_budget, max_pool, C
depth = 1
epsilon = 0.3
min_budget = 20
max_pool = 40
C = 2**0.5


class Node:

    def __init__(self, parent=None, action=None):
        # parent is None or root node
        self.parent = parent
        self.action = action
        self.children = []
        # scores this node get after simulation
        self.evaluations = []
        # off_pool stores all possible children, its max size is max_pool
        self.off_pool = []

    def init_off_pool(self, action_mask):
        '''
        Initialize all possible children. Only root node need this function.
        '''
        global max_pool
        action_mask_1D = np.reshape(action_mask, -1)
        for i in range(action_mask_1D.shape[0]):
            if action_mask_1D[i] == 1:
                self.off_pool.append(Node(self, i))
        if len(self.off_pool) > max_pool:
            random.shuffle(self.off_pool)
            self.off_pool = self.off_pool[:max_pool]

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        '''
        Implementation of UCB.
        '''
        global C
        T = 0
        for child in self.parent.children:
            T += child.num_visits()
        w = np.average(self.evaluations)
        return w + C*(math.log(T)/self.num_visits())**0.5

    def __repr__(self):
        return str(self.action)


def _selection(node):
    '''
    Implementation of selection part of MCTS. Selection depends on UCB score.
    '''
    best_node = None
    for child in node.children:
        if best_node == None or child.score() > best_node.score():
            best_node = child
    return best_node


def _expansion(node):
    '''
    Implementation of expansion part of MCTS. Always select a node from
    off_pool and move it to children.
    '''
    ind = np.random.randint(0, len(node.off_pool))
    res = node.off_pool.pop(ind)
    node.children.append(res)
    return res


def _simulation(game_copy, node, player_id):
    '''
    Implementation of simulation part of MCTS. Step randomly seleced actions
    until the depth is reached.
    '''
    global depth
    obs, rew, done, _ = game_copy.step(action_dict={player_id: node.action})
    for _ in range(depth):
        # assert len(obs.keys()) == 1
        current_player = list(obs.keys())[0]
        # assert current_player == 1 or current_player == -1
        if not done["__all__"]:
            action = random_act(obs)
            obs, rew, done, _ = game_copy.step(
                action_dict={current_player: action})
        else:
            break
    return obs, rew, done


def _backpropagation(obs_pre, obs_curr, node, player_id):
    '''
    Implementation of backpropagation part of MCTS.
    '''
    if obs_pre != None:
        obs_pre = obs_pre[player_id]["partial_observation"]
    obs_curr = obs_curr[player_id]["partial_observation"]
    score = step_score(obs_pre, obs_curr)
    node.visit(score)


def mcts_act(env, obs):
    '''
    Use MCTS to select an action.
    '''
    global min_budget, epsilon
    player_id = list(obs.keys())[0]
    # print("test: ", player_id, env.player)
    # create simulation environment

    # create root node, means the current state
    root_node = Node()
    # initialize off_pool of root_node
    action_mask = obs[player_id]["valid_actions_mask"]
    root_node.init_off_pool(action_mask)
    # print(root_node.off_pool)

    # budget for simulation. Not too big or too small, make sure that MCTS
    # is runtime and action evaluation is reliable.
    budget = max(min_budget, len(root_node.off_pool) * 5)
    for _ in range(budget):
        # MCTS loop
        prob = np.random.rand()
        # print(prob)
        if (prob > epsilon and len(root_node.children) > 0) or \
                len(root_node.off_pool) == 0:
            child_node = _selection(root_node)
        else:
            child_node = _expansion(root_node)
        game_copy = copy.deepcopy(env)
        obs_curr, rew, done = _simulation(game_copy, child_node, player_id)
        _backpropagation(obs, obs_curr, child_node, player_id)
    # return action with highest UCB score
    res = _selection(root_node)
    return res.action
