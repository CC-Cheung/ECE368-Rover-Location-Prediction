import numpy as np

GRID_WIDTH  = 12
GRID_HEIGHT = 8

#-----------------------------------------------------------------------------
# Convention: (x, y) = (0, 0) is the top left of the grid
#
# Each hidden state is encoded as (x, y, action)
# where: 0 <= x <= GRID_WIDTH - 1,
#        0 <= y <= GRID_HEIGHT - 1,
#        action is one of
#        {'left', 'right', 'up', 'down', 'stay'}.
# Note that <action> refers to the *previous* action 
#
# Each observed state is encoded as (x, y)
# where: 0 <= x <= GRID_WIDTH - 1,
#        0 <= y <= GRID_HEIGHT - 1.

class Distribution(dict):
    """
    The Distribution class extend the Python dictionary such that
    each key's value should correspond to the probability of the key.

    Methods
    -------
    renormalize():
      scales all the probabilities so that they sum to 1
    get_mode():
      returns an item with the highest probability, breaking ties arbitrarily
    """
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def renormalize(self):
        normalization_constant = sum(self.values())
        for key in self.keys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.keys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

def get_all_hidden_states():
    # lists all possible hidden states
    all_states = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            possible_prev_actions = ['left', 'right', 'up', 'down', 'stay']

            if x == 0: # previous action could not have been to go right
                possible_prev_actions.remove('right')
            if x == GRID_WIDTH - 1: # could not have gone left
                possible_prev_actions.remove('left')
            if y == 0: # could not have gone down
                possible_prev_actions.remove('down')
            if y == GRID_HEIGHT - 1: # could not have gone up
                possible_prev_actions.remove('up')

            for action in possible_prev_actions:
                all_states.append( (x, y, action) )
    return all_states

def get_all_observed_states():
    # lists all possible observed states
    all_observed_states = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            all_observed_states.append( (x, y) )
    return all_observed_states

def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = Distribution()
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            prior[(x, y, 'stay')] = 1./(GRID_WIDTH*GRID_HEIGHT)
    return prior

def transition_model(state):
    # given a hidden state, return the Distribution for the next hidden state
    x, y, action = state
    next_states  = Distribution()

    # we can always stay where we are
    if action == 'stay':
        next_states[(x, y, 'stay')] = .2
    else:
        next_states[(x, y, 'stay')] = .1

    if y > 0: # we can go up
        if action == 'stay':
            next_states[(x, y-1, 'up')] = .2
        if action == 'up':
            next_states[(x, y-1, 'up')] = .9
    if y < GRID_HEIGHT - 1: # we can go down
        if action == 'stay':
            next_states[(x, y+1, 'down')] = .2
        if action == 'down':
            next_states[(x, y+1, 'down')] = .9
    if x > 0: # we can go left
        if action == 'stay':
            next_states[(x-1, y, 'left')] = .2
        if action == 'left':
            next_states[(x-1, y, 'left')] = .9
    if x < GRID_WIDTH - 1: # we can go right
        if action == 'stay':
            next_states[(x+1, y, 'right')] = .2
        if action == 'right':
            next_states[(x+1, y, 'right')] = .9

    next_states.renormalize()
    return next_states

def observation_model(state):
    # given a hidden state, return the Distribution for its observation
    x, y, action    = state
    observed_states = Distribution()

    radius = 1
    for x_new in range(x - radius, x + radius + 1):
        for y_new in range(y - radius, y + radius + 1):
            if x_new >= 0 and x_new <= GRID_WIDTH - 1 and \
               y_new >= 0 and y_new <= GRID_HEIGHT - 1:
                if (x_new - x)**2 + (y_new - y)**2 <= radius**2:
                    observed_states[(x_new, y_new)] = 1.

    observed_states.renormalize()
    return observed_states


def load_data(filename):
    # loads a list of hidden states and observations saved in txt file
    f = open(filename, 'r')

    hidden_states = []
    observations  = []
    for line in f.readlines():
        line = line.strip()
        if len(line) >= 4:
            parts = line.split()

            hidden_x      = int(parts[0])
            hidden_y      = int(parts[1])
            hidden_action = parts[2]
            hidden_states.append( (hidden_x, hidden_y, hidden_action) )

            if parts[3] == 'missing':
                observations.append(None)
            elif len(parts) == 5:
                observed_x = int(parts[3])
                observed_y = int(parts[4])
                observations.append( (observed_x, observed_y) )

    return hidden_states, observations