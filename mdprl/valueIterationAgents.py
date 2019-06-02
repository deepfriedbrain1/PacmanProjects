# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # set up default values
        negInfinite = -float('inf')
        states = mdp.getStates()
        self.savedValues = self.values.copy()

        # loop through iterations
        for itr in range(iterations):
            # loop through the states
            for state in states:
                possibleActions = mdp.getPossibleActions(state)

                # do while not a terminal state
                if not mdp.isTerminal(state):
                    valueOfAction = negInfinite

                    # loop through the possible actions and get the max Q-value
                    for action in possibleActions:
                        q_value = self.computeQValueFromValues(state, action)
                        valueOfAction = max(q_value, valueOfAction)
                    # save the new max Q-values
                    self.values[state] = valueOfAction
            # update the saved values for later use
            self.savedValues = self.values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # set the default values
        Q_value = 0.0
        discount = self.discount
        reward = 0.0

        # computing the Q-value of actions in the state of saved values with reward, discount and probability
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            transition = self.savedValues[nextState]
            reward = self.mdp.getReward(state, action, nextState)
            Q_value += probability * (reward + (discount * transition))

        # return the Q-value
        return Q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # set the default values
        negInfinite = -float('inf')
        max_q_value = negInfinite
        current_q_value = 0.0
        possibleActions = self.mdp.getPossibleActions(state)
        policy = None

        # loop through possible actions and obtain the policy from the best actions in given state
        for action in possibleActions:
            current_q_value = self.computeQValueFromValues(state, action)

            # compare Q-values and set new policy based on best value/action
            if max_q_value < current_q_value:
                policy = action
                max_q_value = current_q_value

        # return the policy
        return policy

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
