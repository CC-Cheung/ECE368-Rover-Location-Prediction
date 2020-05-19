import numpy as np
import graphics
import rover

def addValToDic(dic,key,val):
    # print(dic,key,val)
    if not key in dic:
        dic[key] = val
    else:
        dic[key] += val
    # print(dic,key,val, "2.")

    return dic
def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    uniform = rover.Distribution()
    uniform.update(dict([(i, 1) for i in all_possible_hidden_states]))
    uniform.renormalize()


    allObs = [all_possible_observed_states if i is None else [i] for i in observations]

    # TODO: Compute the forward messages
    print("forward")
    forward_messages[0]=rover.Distribution()
    for i in prior_distribution:
        obsFactor=observation_model(i)
        for j in obsFactor:
            if j in allObs[0]:
                forward_messages[0][(i[0],i[1], 'stay')]=obsFactor[j]
    forward_messages[0].renormalize()
    # print(0, ". ", forward_messages[0])

    for i in range(num_time_steps - 1):
        forward_messages[i + 1] = rover.Distribution()
        for j in forward_messages[i]:  # states of i
            transTemp = rover.transition_model(j)
            for k in transTemp:
                # print(k, allObs[i+1])
                obsFactor=observation_model(k) #(place): chance
                for l in obsFactor:
                    if l in allObs[i+1]:
                        forward_messages[i + 1] = addValToDic(forward_messages[i + 1], k,
                                                               forward_messages[i][j] * transTemp[k]*obsFactor[l])

        forward_messages[i + 1].renormalize()
        # print(i, ". ", forward_messages[i])
        # print(i + 1, ". ", allObs[i + 1], observations[i + 1])
        # print(i + 1, ". ", forward_messages[i + 1], "\n")
    # TODO: Compute the backward messages

    backward_messages[-1] = uniform

    print("back")

    # print(-1, ". ", backward_messages[-1])
    for i in range(num_time_steps - 1, 0, -1):
        backward_messages[i - 1] = rover.Distribution()
        # revTrans=rover.Distribution()
        for j in all_possible_hidden_states: #states of i-1
            transTemp = rover.transition_model(j)
            for k in transTemp:
                if k in backward_messages[i]:
                    obsFactor=observation_model(k)
                    for l in obsFactor:
                        if l in allObs[i]:#
                            backward_messages[i-1]=addValToDic(backward_messages[i-1], j, backward_messages[i][k]*transTemp[k]*obsFactor[l])


        backward_messages[i - 1].renormalize()
        # print(i, ". ", backward_messages[i].get_mode(), backward_messages[i][backward_messages[i].get_mode()],
        #       "message", backward_messages[i])
        # print(i, ".", allObs[i], observations[i])
        # print(i - 1, ". ", backward_messages[i - 1].get_mode(),
        #       backward_messages[i - 1][backward_messages[i - 1].get_mode()], "message", backward_messages[i - 1], '\n')

    # TODO: Compute the marginals
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        marginals[i].update(dict(
            [(k, forward_messages[i][j] * backward_messages[i][k]) for j in forward_messages[i] for k in
             backward_messages[i] if j == k]))
        marginals[i].renormalize()
        # print(marginals[i], forward_messages[i], backward_messages[i])
    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    maxLastState = []
    maxLast = -100000000
    allObs = [all_possible_observed_states if i is None else [i] for i in observations]

    w = [None] * num_time_steps
    estimated_hidden_states = [rover.Distribution()] * num_time_steps

    w[0] = rover.Distribution()  # assume stay

    for i in prior_distribution:
        obsFactor=observation_model(i)
        for j in obsFactor:
            if j in allObs[0]:
                w[0][i]=["no previous", np.log(prior_distribution[i])*np.log(obsFactor[j])]


    for i in range(num_time_steps- 1):
        w[i + 1] = rover.Distribution()
        trans = rover.Distribution()

        for j in w[i]:
            tempTrans = transition_model(j)
            for k in tempTrans: #of n+1
                # print(w[i][j])

                maybeMax=np.log(tempTrans[k]) + w[i][j][1]

                if k in w[i + 1]:
                    newMax = np.maximum(maybeMax, w[i + 1][k][1])
                    if newMax != w[i + 1][k][1]:
                        trans[k] = [j, newMax] #newMax= np.maximum(np.log(tempTrans[k]) + w[i][j][1], w[i + 1][k][1])
                else:
                    trans[k] = [j, maybeMax]


        for j in trans:
            obsFactor=observation_model(j)
            for k in obsFactor:
                if k in allObs[i+1]:
                    w[i + 1][j] = [trans[j][0], trans[j][1] + np.log(obsFactor[k])]
        # print(i, w[i], "transitions")
        # print (i+1,observations[i+1],w[i+1], "\n")

    # for i in range(100):
    #     print(i, ". ", w[i])
    for i in w[-1]:
        if w[-1][i][1] > maxLast:
            maxLast = w[-1][i][1]
            maxLastState = i
    estimated_hidden_states[-1] = maxLastState
    for i in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[i] = w[i + 1][estimated_hidden_states[i + 1]][0]
    # print (estimated_hidden_states[-1])
    # TODO: Write your code here

    return estimated_hidden_states

def illegalMove(state1, state2):
    return not state2 in rover.transition_model(state1)
if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True

    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    start = 25
    end = 49
    start = 0
    end = 100
    observations = observations[start:end]
    num_time_steps = len(hidden_states)

    all_possible_observed_states = rover.get_all_observed_states()
    all_possible_hidden_states = rover.get_all_hidden_states()
    prior_distribution = rover.initial_distribution()
    #
    # print('Running forward-backward...')
    # print(observations)
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    # print('\n---')
    # for i in range(len(marginals)):
    #     print(i + start, marginals[i], observations[i])

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    marginalStates=[marginals[i].get_mode() for i in range(num_time_steps)]
    # print(marginalStates)
    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    margError=1-np.sum([marginalStates[i]==hidden_states[i] for i in range(num_time_steps)])/num_time_steps
    viterbError = 1 - np.sum([estimated_states[i] == hidden_states[i] for i in range(num_time_steps)]) / num_time_steps
    print("marginal error=", margError, "viterbi error=", viterbError)
    margIllegal=[(i, marginalStates[i], marginalStates[i+1]) for i in range(num_time_steps-1) if illegalMove(marginalStates[i], marginalStates[i+1])]
    print (margIllegal)
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

