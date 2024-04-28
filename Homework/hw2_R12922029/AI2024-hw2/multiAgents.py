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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # C = 0;
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates() 
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ## Idea : 
        # When the ghosts are scared, eat food and ghost
        # When the ghosts are not scared, eat food and avoid ghost 
        FoodList = newFood.asList()
        FoodDistanceList = [manhattanDistance(newPos, food) for food in FoodList]
        
        GhostList = [ghost.getPosition() for ghost in newGhostStates]
        GhostDistanceList = [manhattanDistance(newPos, ghost) for ghost in GhostList]
        ClosestGhostDistance = float(min(GhostDistanceList))
        
        # handle the case when there is no food left
        if len(FoodDistanceList) == 0:
            ClosestFoodDistance = float('inf')
        else:
            ClosestFoodDistance = float(min(FoodDistanceList))
        
        haunt_ghost = 1;
        for i in range(len(newScaredTimes)):
            if newScaredTimes[i] == 0:
                haunt_ghost = 0;
                break;
        for i in range(len(GhostDistanceList)):
            if GhostDistanceList[i] < 2: 
                danger = - float('inf');
            else:
                danger = float(0);

        if haunt_ghost == 1:
            # print("haunt_ghost")
            # C += 1
            # print(C)
            return successorGameState.getScore() + 1.0/ClosestFoodDistance + 10.0/ClosestGhostDistance
        else: 
            # C += 1;
            # print(C)
            return successorGameState.getScore() + 1.0/ClosestFoodDistance + danger

def scoreEvaluationFunction(currentGameState: GameState):
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
    
    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()       

        return self.maxPoint(gameState, 0, 0)[0] # return the best action for pacman

    def minimax(self, gameState, agentIndex, depth):  
        # if agentIndex > 2 or depth > 9:
        #     input()
        
        # reach depth limit of terminal state
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():  
            return self.evaluationFunction(gameState)
        # pacman
        elif agentIndex is 0:
            return self.maxPoint(gameState, agentIndex, depth)[1]
        # ghost
        else: 
            return self.minPoint(gameState, agentIndex, depth)[1]

    def maxPoint(self, gameState, agentIndex, depth):
        maxPoint = ("max", -float("inf"))  
        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex,action)
            nextAgentIndex = (depth + 1) % gameState.getNumAgents()
            succAction = (action,self.minimax(nextGameState, nextAgentIndex, depth+1))  
            maxPoint = max(maxPoint,succAction,key=lambda x:x[1])
        return maxPoint

    def minPoint(self, gameState, agentIndex, depth):
        minPoint = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex,action)
            nextAgentIndex = (depth + 1) % gameState.getNumAgents()
            succAction = (action,self.minimax(nextGameState, nextAgentIndex, depth+1)) 
            minPoint = min(minPoint,succAction,key=lambda x:x[1])
        return minPoint

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.maxPoint(gameState, 0, 0, -float("inf"), float("inf"))[0] # return the best action for pacman

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):  
        # reach depth limit of terminal state
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin(): 
            return self.evaluationFunction(gameState)
        # pacman
        elif agentIndex is 0:  
            return self.maxPoint(gameState, agentIndex, depth, alpha, beta)[1]
        # ghost
        else: 
            return self.minPoint(gameState, agentIndex, depth, alpha, beta)[1]

    def maxPoint(self, gameState, agentIndex, depth, alpha, beta):
        maxPoint = ("max", -float("inf"))  
        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex,action)
            nextAgentIndex = (depth + 1) % gameState.getNumAgents()
            succAction = (action,self.alphabeta(nextGameState, nextAgentIndex, depth+1, alpha, beta))  
            maxPoint = max(maxPoint,succAction,key=lambda x:x[1])
            # pruning
            if maxPoint[1] > beta:
                return maxPoint
            else:
                alpha = max(alpha, maxPoint[1])
        return maxPoint

    def minPoint(self, gameState, agentIndex, depth, alpha, beta):
        minPoint = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex,action)
            nextAgentIndex = (depth + 1) % gameState.getNumAgents()
            succAction = (action,self.alphabeta(nextGameState, nextAgentIndex, depth+1, alpha, beta)) 
            minPoint = min(minPoint,succAction,key=lambda x:x[1])
            # pruning
            if minPoint[1] < alpha:
                return minPoint
            else:
                beta = min(beta, minPoint[1])
        return minPoint

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        # return self.expectimax(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    # return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
