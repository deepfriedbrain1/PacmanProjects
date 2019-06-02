# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghosts = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in ghosts]
        nearestGhost = min(ghostDistances)
        foodLocations = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in foodLocations]
        if len(foodDistances) == 0:
            return 1000
        nearestFood = min(foodDistances)
        dontStop = 0
        if action == "Stop":
            dontStop -= 100
        return successorGameState.getScore() + dontStop + (0.8 * nearestGhost) / (1.20 * nearestFood)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Return action list, agent is pac-man by default (no parameter)
        return self.value(gameState)[1]

    def value(self, gameState, depth=0, agent=0):

        # base case: depth reached or no more legal moves
        if self.depth == depth or len(gameState.getLegalActions(agent)) < 1:
            return gameState.getScore(), ""
        # pac-man agent == 0
        if agent < 1:
            return self.max_value(gameState, depth)
        # ghost agents >= 1
        elif agent >= 1:
            return self.min_value(gameState, depth, agent)

    def max_value(self, gameState, depth, agent=0):
        # set default value
        v = float("-inf")
        return self.getValueAction(gameState, depth, agent, v, "max")

    def min_value(self, gameState, depth, agent):
        # set default value
        v = float("inf")
        return self.getValueAction(gameState, depth, agent, v, "min")

    def getValueAction(self, gameState, depth, agent, initialValue, type):
        # set default variables
        actions = gameState.getLegalActions(agent)
        action = ""
        selectionType = type
        v = initialValue

        for move in actions:
            successor = gameState.generateSuccessor(agent, move)
            nextAgent = agent + 1
            nextDepth = depth

            # if all agents have moved and reset to pac-man and update the depth
            if nextAgent == gameState.getNumAgents():
                nextDepth += 1
                nextAgent = 0

            this_value = self.value(successor, nextDepth, nextAgent)[0]

            if selectionType == "min" and this_value < v:
                v = min(v, this_value)
                action = move

            if selectionType == "max" and this_value > v:
                v = max(v, this_value)
                action = move

        return v, action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.bestOptionOnPath(gameState)[0]
        util.raiseNotDefined()

    def bestOptionOnPath(self, gameState, alpha=float("-inf"), beta=float("inf"), depth=0, agent=0):

        # base case: depth reached or no more legal moves
        if depth == self.depth or len(gameState.getLegalActions(agent)) < 1:
            return "", gameState.getScore()

        if agent < 1:
            v = float("-inf")
            return self.max_value(gameState, alpha, beta, v, depth, "max")
        elif agent >= 1:
            v = float("inf")
            return self.min_value(gameState, alpha, beta, v, depth, "min", agent)

        util.raiseNotDefined()

    def max_value(self, gameState, alpha, beta, initialValue, depth, type, agent=0):
        return self.getUtility(gameState, alpha, beta, initialValue, depth, type, agent)

    def min_value(self, gameState, alpha, beta, initialValue, depth, type, agent):
        return self.getUtility(gameState, alpha, beta, initialValue, depth, type, agent)

    def getUtility(self, gameState, alpha, beta, v, depth, type, agent):
        # set default values
        actions = gameState.getLegalActions(agent)
        action = ""
        selectionType = type

        for move in actions:
            nextAgent = 1 + agent
            nextDepth = depth
            successor = gameState.generateSuccessor(agent, move)

            # all agents processed reset to pac-man and update the depth
            if nextAgent == gameState.getNumAgents():
                nextDepth += 1
                nextAgent = 0

            this_value = self.bestOptionOnPath(successor, alpha, beta, nextDepth, nextAgent)[1]

            if selectionType == "max" and this_value > v :
                v = max(v, this_value)
                action = move
                alpha = max(alpha, v)

                if v > beta:
                    return action, v

            if selectionType == "min" and this_value < v:
                v = min(v, this_value)
                action = move
                beta = min(beta, v)

                if v < alpha:
                    return action, v

        return action, v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return  self.value(gameState)[0]

    def value(self, gameState, depth=0, agent=0):

        # base case: depth reached or no more legal moves
        if depth == self.depth or len(gameState.getLegalActions(agent)) < 1:
            return "", self.evaluationFunction(gameState)

        if agent < 1:
            return self.max_value(gameState, depth)
        elif agent >= 1:
            return self.exp_value(gameState, depth, agent)

    def max_value(self, gameState, depth, agent=0):
        # set default variables
        action = ""
        v = float("-inf")
        actions = gameState.getLegalActions(agent)

        for move in actions:
            nextAgent = 1 + agent
            nextDepth = depth
            successor = gameState.generateSuccessor(agent, move)

            # all agents processed, reset to pac-man and update the depth
            if nextAgent == gameState.getNumAgents():
                nextDepth += 1
                nextAgent = 0

            this_value = self.value(successor, nextDepth, nextAgent)[1]

            if v < this_value:
                action = move

            v = max(v, this_value)

        return action, v

    def exp_value(self, gameState, depth, agent):
        # set default variables
        action = ""
        actions = gameState.getLegalActions(agent)
        totalActions = len(actions)
        v = 0
        p = (1.0 / totalActions) # p = probability(successors), uniformed distribution

        for move in actions:
            nextAgent = 1 + agent
            nextDepth = depth
            successor = gameState.generateSuccessor(agent, move)

            if nextAgent == gameState.getNumAgents():
                nextDepth += 1
                nextAgent = 0

            this_value = self.value(successor, nextDepth, nextAgent)[1]

            # v += p * value(successor)
            v += p * this_value

        return action, v

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Using the ratio of 15 : nearest food dot to encourage pac-man to
      move to towards the food dot. Also using the current game score to
      encourage pac-man to maxmize his movements and finish the game.
      The remaining capsules, food and nearest ghost are also used to
      ecourage pac-man to eat the capsule and food as they have a negative
      impact on the overall value being returned. The nearest ghost has a
      negative weight so that pac-man is a little more aggressive towards
      closing the gap between the ghost.
    """
    "*** YOUR CODE HERE ***"
    # set the variables to use in the betterEvaluationFunction
    nearestFood = 1
    food = currentGameState.getFood()
    foodLocations = food.asList()
    remainingFood = len(foodLocations)
    pacman = currentGameState.getPacmanPosition()
    pacmans_distance_to_foods = [manhattanDistance(pacman, food) for food in foodLocations]
    ghosts = currentGameState.getGhostPositions()
    pacman_distance_to_ghost = [manhattanDistance(pacman, ghost) for ghost in ghosts]
    nearestGhost = min(pacman_distance_to_ghost)
    score = currentGameState.getScore()
    capsuleLocations = currentGameState.getCapsules()
    remainingCapsules = len(capsuleLocations)

    if nearestGhost <= 1:
        nearestFood = 9999

    if len(capsuleLocations) > 0:
        pacman_distance_to_capsule = min([manhattanDistance(pacman, nearestCapsule) for nearestCapsule in capsuleLocations])
    else:
        pacman_distance_to_capsule = 0

    if remainingFood >= 1:
        nearestFood = min(pacmans_distance_to_foods)

    return (15.0/nearestFood) + (105*score) + (-30*remainingCapsules) + (-40*remainingFood) + (-2*nearestGhost) \
           + (-pacman_distance_to_capsule)

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

