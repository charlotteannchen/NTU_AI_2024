"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
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
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    ## All of your search functions need to return a list of actions that will lead the
    # agent from the start to the goal. These actions all have to be legal moves (valid directions, no
    # moving through walls).
    # List of actions: [‘South’, ‘South’, ‘West’, ‘South’, ‘West’, ‘West’]
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    ## implementation a depth first search algo to solve pacman
    frontier = util.Stack() # initialize the frontier as a stack
    exploredSet = [] # list that records explored nodes
    frontier.push((problem.getStartState(), [])) # push the start node to the frontier
    while not frontier.isEmpty(): 
        currState, actions = frontier.pop() # pop the last node from the frontier
        # print("Current State:", currState, "Actions:", actions, "Current Cost:", currCost)
        
        if problem.isGoalState(currState):
                return actions
            
        if currState not in exploredSet: 
            exploredSet.append(currState) # add the current state to the explored set
            for succState, succAction, _ in problem.getSuccessors(currState):
                frontier.push((succState, actions + [succAction]))
        # print(actions)

    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue() # initialize the frontier as a queue
    exploredSet = []
    frontier.push((problem.getStartState(), []))
    while not frontier.isEmpty():
        currState, actions= frontier.pop() # begin exploring first (earliest-pushed) node on frontier
        
        if problem.isGoalState(currState):
                return actions
            
        if currState not in exploredSet:
            exploredSet.append(currState) 
            for succState, succAction, _ in problem.getSuccessors(currState):
                frontier.push((succState, actions + [succAction]))
    return []
    # util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue() # initialize the frontier as a priority queue
    exploredSet = {}
    frontier.push((problem.getStartState(), [], 0), 0) # (state, actions, cost, real cost + heuristic) 
    while not frontier.isEmpty():
        currState, actions, currCost = frontier.pop() # begin exploring first (lowest-cost) node on frontier

        if problem.isGoalState(currState):
            return actions

        if (currState not in exploredSet) or (currCost < exploredSet[currState]):
            exploredSet[currState] = currCost

            for succState, succAction, succCost in problem.getSuccessors(currState):
                if succState not in exploredSet:
                    newCost = currCost + succCost
                    frontier.update((succState, actions + [succAction], newCost), newCost)
    return []
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue() # initialize the frontier as a priority queue
    exploredSet = [] 
    frontier.push((problem.getStartState(), [], 0), 0) # (state, actions, cost, real cost + heuristic)

    while not frontier.isEmpty():
        currState, actions, currCost = frontier.pop() # begin exploring first (lowest-combined (cost+heuristic) ) node on frontier

        exploredSet.append((currState, currCost))

        if problem.isGoalState(currState):
            return actions
        else:
            for succState, succAction, _ in problem.getSuccessors(currState):
                newActions = actions + [succAction]
                newCost = problem.getCostOfActions(newActions)
                newNode = (succState, newActions, newCost)
                
                already_explored = False
                for explored in exploredSet:
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                if not already_explored:
                    frontier.push(newNode, newCost + heuristic(succState, problem))
                    exploredSet.append((succState, newCost))

    return []

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
