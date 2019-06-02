# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # Utilize stack with a generic search algorithm to implement DFS
    return genericSearch(problem, "stack")
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Utilize a queue with a generic search algorithm to implement BFS
    return genericSearch(problem, "queue")
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Utilize A* search with no heuristic (null)
    return aStarSearch(problem, nullHeuristic)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Utilize a priority queue with a generic search algorithm to implement A* Search
    return genericSearch(problem, "priority_queue", heuristic)
    util.raiseNotDefined()

def genericSearch(problem, type, heuristic=None):

    # clear out compiler warnings and set default values
    actions = []
    frontier = None

    # initial state of the problem
    initialState = problem.getStartState()

    # list of visited nodes (initialize the explored set to be empty)
    visitedVertex = []

    # list of actions
    currentActions = []

    # set type of data structure to use and initialize the frontier using initial node
    if type == "stack":
        frontier = util.Stack()
        frontier.push((initialState, currentActions))

    if type == "queue":
        frontier = util.Queue()
        frontier.push((initialState, currentActions))

    if type == "priority_queue":
        frontier = util.PriorityQueue()
        frontier.push((initialState, currentActions), heuristic(initialState, problem))

    # loop through frontier while not empty, else return the empty list (actions)
    while frontier:
        # obtain the first vertex and actions
        vertex, actions = frontier.pop()

        # if the vertex is the GOAL return the current actions list
        if problem.isGoalState(vertex):
            return actions

        # if the vertex has not been visited yet, add to visited list
        if vertex not in visitedVertex:
            visitedVertex.append(vertex)
            # obtain the next set of actions from current node
            nextActions = problem.getSuccessors(vertex)
            # for each action available in the current node's path
            for nextVertexActions in nextActions:
                # obtain the vertex, heading and cost of that action
                nextVertex, heading, cost = nextVertexActions
                # add the current heading with the new heading
                act = actions + [heading]
                # push the new frontier nodes into appropriate data structures
                if (type == "queue") or (type == "stack"):
                    frontier.push((nextVertex, act))
                # priority queue implementation strategy utilizes heuristic to add total costs
                if type == "priority_queue":
                    currentCost = problem.getCostOfActions(act) + heuristic(nextVertex, problem)
                    frontier.push((nextVertex, act), currentCost)

    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
