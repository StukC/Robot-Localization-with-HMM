import numpy as np

# Chase Stuk #8942 8283
# Date Created: 3/22/2022
# Date Last Modified: 3/26/2022

height = 6
width = 7
obstacles = ((1, 2),
             (1, 3),
             (1, 4),
             (1, 5),
             (2, 2),
             (2, 5),
             (3, 1),
             (3, 2),
             (3, 5),
             (4, 4),
             (4, 5))
# E = Empty, # = Obstacle
#   0 1 2 3 4 5 6
# 0 E E E E E E E
# 1 E E # # # # E
# 2 E E # E E # E
# 3 E # # E E # E
# 4 E E E E # # E
# 5 E E E E E E E
wall_prob = 0.85
wall_prob_fail = 0.15
open_prob = 0.9
open_prob_fail = 0.1
forward_prob = 0.8
drift_prob = 0.1
sequence_item = 'evidence'

NORTH = 1
EAST = 2

# robot will perform following sequence
sequence = [[0, 0, 0, 1],
            NORTH,
            [0, 1, 0, 0],
            EAST,
            [0, 1, 0, 0],
            EAST,
            [0, 0, 1, 0]]

# places open spaces
open_spaces = [(x, y) for x in range(6) for y in range(7) if (x, y) not in obstacles]


# move in direction
def moveit(location, move):
    global new_location
    if move == 0:
        new_location = (location[0], location[1] - 1)
    elif move == 1:
        new_location = (location[0] - 1, location[1])
    elif move == 2:
        new_location = (location[0], location[1] + 1)
    elif move == 3:
        new_location = (location[0] + 1, location[1])
        # if new_location is out of bounds, return original location
    if (new_location in obstacles  # move into obstacle
            or new_location[0] <= -1 or new_location[0] >= height  # up or down
            or new_location[1] <= -1 or new_location[1] >= width):  # left or right
        return location[0], location[1]  # don't move
    else:
        return new_location  # new location


# returns probability
def evidence_probability(evidence, location):
    prob = 1  # prob to return
    wall_here = []  # wall any direction

    # iterate directions
    for move in range(4):
        if location == moveit(location, move):  # if wall hit
            wall_here.append(1)  # wall found
        else:
            wall_here.append(0)  # open space

    # iterate directions
    for num in range(4):
        if wall_here[num] == 1 and evidence[num] == 1:  # wall found
            prob *= wall_prob
        elif wall_here[num] == 0 and evidence[num] == 0:  # open space found
            prob *= open_prob
        elif wall_here[num] == 0 and evidence[num] == 1:  # mistakenly found wall
            prob *= open_prob_fail
        elif wall_here[num] == 1 and evidence[num] == 0:  # mistakenly found open space
            prob *= wall_prob_fail

    return prob  # return result


def transitional_probability(move, action):
    # move in intended direction
    move_forward = moveit(move, action)

    # drift left
    move_counter_clockwise = moveit(move, (action - 1) % 4)

    # drift right
    state_clockwise = moveit(move, (action + 1) % 4)

    # returns grouped probabilities for current states
    return ((move_forward, forward_prob),
            (move_counter_clockwise, drift_prob),
            (state_clockwise, drift_prob))


# displays distribution
def display(distribution):
    for row in distribution:
        for cell in row:
            if cell < 0.0000000001:
                print('####    ', end='')
            else:  # print out probability
                if len("{prob:.2f}".format(prob=cell * 100)) == 7:  # if probability is less than 1%
                    print("{prob:.2f}   ".format(prob=cell * 100), end='')  # print out probability
                else:
                    print("{prob:.2f}    ".format(prob=cell * 100), end='')
        print()


# sensing update
def filtering(action, evidence):
    for os in open_spaces:  # iterate through open spaces
        action[os[0], os[1]] *= evidence_probability(evidence, os)  # calculate
    action /= np.sum(action)  # returns P(S | Si, a)


# motion update
def prediction(num, action):
    new_num = np.zeros((height, width), np.float64)  # numpy array of zeros
    for os in open_spaces:  # iterate though open spaces
        for (state, prob) in transitional_probability(os, action):  # iterate through states
            new_num[state[0], state[1]] += prob * num[os[0], os[1]]  # add on term for total probability
    return new_num  # update distribution


# backward pass
def backward(dist, evidence, action):
    new_dist = np.zeros((height, width), np.float64)
    for os in open_spaces:
        for (state, prob) in transitional_probability(os, action):
            # add on term for total probability
            # value += recursive probability * evidence prob * transition prob
            new_dist[os[0], os[1]] += dist[state[0], state[1]] * evidence_probability(evidence, state) * prob
    # normalize
    new_dist /= np.sum(new_dist)
    return new_dist


# distribution initialization
distribution = []
forward_evidence = []
forward_dist = []
actions = []

# backward distribution probabilities
backwards_distribution = [np.ones((height, width), np.float64)]

# create list
for _ in range(6):
    distribution.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # append list

distribution = np.array(distribution)  # create numpy array
initial_probability = 1.0 / len(open_spaces)  # equal likelihood

# iterate through open spaces
for os in open_spaces:
    distribution[os[0], os[1]] = initial_probability  # set equal likelihood

print('Initial Location Probabilities')
display(distribution)
print()

# sequence process until empty
while len(sequence) != 0:
    if sequence_item == 'evidence':
        sequence_item = 'action'
        evidence = sequence.pop(0)
        filtering(distribution, evidence)
        print('Filtering after Evidence ' + str(evidence))
        display(distribution)

        # copy distribution
        new_dist = np.zeros((height, width), np.float64)
        for os in open_spaces:
            new_dist[os[0], os[1]] = distribution[os[0], os[1]]
        forward_dist.append(new_dist)  # append distribution
        forward_evidence.append(evidence)  # append evidence
    else:
        sequence_item = 'evidence'
        num = sequence.pop(0)
        distribution = prediction(distribution, num)
        print('Prediction after Action ' + ('E' if num == 2 else 'N'))  # print action
        display(distribution)
        actions.append(num)

    print()

for i in range(len(forward_dist) - 2, -1, -1):
    next_backward_dist = backward(backwards_distribution[0], forward_evidence[i + 1], actions[i])  # backward pass
    backwards_distribution.insert(0, next_backward_dist)  # insert distribution
    smooth = np.multiply(forward_dist[i], next_backward_dist)  # smooth
    smooth /= np.sum(smooth)
