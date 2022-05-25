import math
import copy
import random
import numpy as np

from scorer import step_score
from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions

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


def _simulation(game_copy):
    '''
    Implementation of simulation part of MCTS. Step randomly seleced actions
    until the depth is reached.
    '''
    global depth
    for _ in range(depth):
        flag = True
        while True:
            if len(game_copy.get_available_actions()) == 0:
                flag = False
                break
            action_choice = self.rnd.choice(
                game_copy.get_available_actions())
            if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
                break
        if flag:
            position = self.rnd.choice(action_choice.positions) if len(
                action_choice.positions) > 0 else None
            player = self.rnd.choice(action_choice.players) if len(
                action_choice.players) > 0 else None
            action = botbowl.Action(action_choice.action_type,
                                    position=position, player=player)
            game_copy.step(action)
        else:
            break


def _backpropagation(obs_pre, obs_curr, node):
    '''
    Implementation of backpropagation part of MCTS.
    '''
    score = step_score(obs_pre, obs_curr)
    node.visit(score)


def act(env, obs, player_id):
    '''
    Use MCTS to select an action.
    '''
    global min_budget, epsilon
    # create simulation environment
    game_copy = copy.deepcopy(env)

    # create root node, means the current state
    root_node = Node()
    # initialize off_pool of root_node
    action_mask = obs[player_id]["valid_actions_mask"]
    root_node.init_off_pool(action_mask)

    # budget for simulation. Not too big or too small, make sure that MCTS
    # is runtime and action evaluation is reliable.
    budget = max(min_budget, len(root_node.off_pool) * 5)
    for _ in range(budget):
        # MCTS loop
        prob = np.random.rand()
        if (prob > epsilon and len(root_node.children) > 0) or \
                len(root_node.off_pool) == 0:
            child_node = _selection(root_node)
        else:
            child_node = _expansion(root_node)
        obs_curr = _simulation(game_copy, child_node)
        _backpropagation(obs, obs_curr, child_node)
    # return action with highest UCB score
    res = _selection(root_node)
    return res.action
