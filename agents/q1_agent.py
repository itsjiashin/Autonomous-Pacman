# PacmanValueIterationAgent.py
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


import util

from agents.learningAgents import ValueEstimationAgent
from game import Grid, Actions, Directions
import math
from pacman import GameState
import random
import numpy as np


class Q1Agent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        Q1 agent is a ValueIterationAgent takes a Markov decision process
        (see pacmanMDP.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp="PacmanMDP", discount=0.6, iterations=500, pretrained_values=None):
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
        mdp_func = util.import_by_name('./', mdp)
        self.mdp_func = mdp_func

        print('[Q1Agent] using mdp ' + mdp_func.__name__)

        self.discount = float(discount)
        self.iterations = int(iterations)

        if pretrained_values:
            self.values = np.loadtxt(pretrained_values)
        else:
            self.values = None

    ########################################################################
    ####            CODE FOR YOU TO MODIFY STARTS HERE                  ####
    ########################################################################

    def registerInitialState(self, state: GameState):

        # set up the mdp with the agent starting state
        self.MDP = self.mdp_func(state)

        # if we haven't solved the mdp yet or are not using pretrained weights
        if self.values is None:

            print("solving MDP")
            possible_states = self.MDP.getStates()
            self.values = np.zeros((self.MDP.grid_width, self.MDP.grid_height))

            # Write value iteration code here
            "*** YOUR CODE STARTS HERE ***"
            for _ in range(self.iterations):
                #https://numpy.org/doc/stable/reference/generated/numpy.copy.html
                new_values = np.copy(self.values)
                for mdp_state in possible_states:
                    #If terminal state, skip over
                    if self.MDP.isTerminal(mdp_state):
                        continue
                    max_value = float('-inf')
                    for action in self.MDP.getPossibleActions(mdp_state):
                        q_value = self.computeQValueFromValues(mdp_state, action)
                        max_value = max(max_value, q_value)
                    #Associate the max_value to the curent mdp state
                    new_values[mdp_state[0]][mdp_state[1]] = max_value
                self.values = new_values 
            "*** YOUR CODE ENDS HERE ***"

            np.savetxt(f"./logs/{state.data.layout.layoutFileName[:-4]}.model", self.values,
                       header=f"{{'discount': {self.discount}, 'iterations': {self.iterations}}}")

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        #Following the lecture slides of V_i+1(s) = max of a over the sum of s' over T(s, a, s')[R(s, a, s') + γV_i(s')]
        for next_state, prob in self.MDP.getTransitionStatesAndProbs(state, action):
            q_val += prob * (self.MDP.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        "*** YOUR CODE HERE ***"
        best_action = None
        max_val = float('-inf')
        #Loop through all possible actions of the current state to find out which action results in the highest q-value 
        for action in self.MDP.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_val:
                max_val = q_value 
                best_action = action 
        return best_action
    ########################################################################
    ####            CODE FOR YOU TO MODIFY ENDS HERE                    ####
    ########################################################################

    def getValue(self, state):
        """
        Takes an (x,y) tuple and returns the value of the state (computed in __init__).
        """
        return self.values[state[0], state[1]]

    def getPolicy(self, state):
        pacman_loc = state.getPacmanPosition()
        return self.computeActionFromValues(pacman_loc)

    def getAction(self, state: GameState):
        "Returns the policy at the state "

        pacman_location = state.getPacmanPosition()
        if self.MDP.isTerminal(pacman_location):
            raise util.ReachedTerminalStateException("Reached a Terminal State")
        else:
            best_action = self.getPolicy(state)
            return self.MDP.apply_noise_to_action(pacman_location, best_action)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


