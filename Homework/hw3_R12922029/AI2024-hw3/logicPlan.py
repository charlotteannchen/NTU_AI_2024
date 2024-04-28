# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    sentence1_1 = A | B
    sentence1_2 = (~A % (~B | C))
    sentence1_3 = disjoin(~A, ~B, C)
    return conjoin([sentence1_1, sentence1_2, sentence1_3])
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    A = PropSymbolExpr('A')
    B = PropSymbolExpr('B')
    C = PropSymbolExpr('C')
    D = PropSymbolExpr('D')
    sentence2_1 = C % (B | D)
    sentence2_2 = A >> (~B & ~D)
    sentence2_3 = (~(B & ~C)) >> A
    sentence2_4 = ~D >> C
    return conjoin([sentence2_1, sentence2_2, sentence2_3, sentence2_4])
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # A = Expr('PacmanAlive_1')
    # B = Expr('PacmanAlive_0')
    # C = Expr('PacmanBorn_0')
    # D = Expr('PacmanKilled_0')
    
    A = Expr('PacmanAlive_0')
    B = Expr('PacmanAlive_1')
    C = Expr('PacmanBorn_0')
    D = Expr('PacmanKilled_0')
    sentence3_1 = B % ((A & ~D) | (~A & C)) 
    sentence3_2 = ~(A & C)
    sentence3_3 = C
    return conjoin([sentence3_1, sentence3_2, sentence3_3])

    # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    # print(pycoSAT(cnf_sentence))
    return pycoSAT(cnf_sentence)

def findModelUnderstandingCheck() -> Dict[Expr, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    """
    a = Expr('A')
    "*** BEGIN YOUR CODE HERE ***"
    # print("a.__dict__ is:", a.__dict__) # might be helpful for getting ideas
    class SimulatedfindModel:
        def __init__(self, variable_name: str = 'A'):
            self.set_variable_name(variable_name)
            
        def set_variable_name(self, value):
            self.variable_name = value
            
        def __repr__(self):
            return self.variable_name
    if a.__dict__['op'] == 'A':
        return {SimulatedfindModel('a'): True}
    else:
        return False
    return {SimulatedfindModel('a'): True}
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # Returns True if the premise entails the conclusion and False otherwise.
    entailments = premise & ~conclusion
    if(findModel(entailments) == False):
        return True
    else:
        return False
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # print("Inverse_statement: ", inverse_statement, "\nAssignments: ", assignments)
    # print("Returning: ", pl_true(~inverse_statement, assignments))
    return pl_true(~inverse_statement, assignments)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    # print("Literals: ", literals)
    # print("atLeastOne: ", disjoin(literals))
    return disjoin(literals)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"
    result = []
    for i in range(len(literals)):
        for j in range(i+1, len(literals)):
            result.append(~literals[i] | ~literals[j])
    # print("Literals: ", literals)
    # print("atMostOne: ", conjoin(result))
    return conjoin(result)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # print("Literals: ", literals)
    # print("exactlyOne: ", conjoin([atLeastOne(literals), atMostOne(literals)]))
    return conjoin([atLeastOne(literals), atMostOne(literals)])
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last)) 
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None
    # print("Possible Causes: ", possible_causes)
    "*** BEGIN YOUR CODE HERE ***"
    ## 檢查現在位置和所有可能的原因之間是否具有雙向關係(biconditional equivalence)
    return PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes) # disjoin : OR, conjoin : AND, % : biconditional
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    # ~P[x,y]_t-1 & ~W[x,y] & (P[x,y+1]_t-1 & South_t-1 | ...)
    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)]) 
    
    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)
    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])
    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"
    
    # - for all (x, y) in all_coords:
    #         If a wall is at (x, y) --> Pacman is not at (x, y)
    possiblePositions = []  
    for (x,y) in non_outer_wall_coords:
        
        possiblePositions.append(PropSymbolExpr(pacman_str, x, y, time = t)) # all the possible positions that pacman can be at time = t
    for (x,y) in all_coords:
        # Why not using biconditional here? because we only need to check if there is a wall at (x,y) then pacman is not at (x,y)
        # If there is no wall at (x,y), we don't need to check if pacman is at (x,y) then there is no wall at (x,y)
        
        pacphysics_sentences.append(PropSymbolExpr(wall_str, x, y) >> ~(PropSymbolExpr(pacman_str, x, y, time=t)))   
    
    # - Pacman is at exactly one of the squares at timestep t.
    pacphysics_sentences.append(exactlyOne(possiblePositions)) # Make sure pacman is at exactly one of the locations at timestep t
    
    possibleActions = []  
    for action in DIRECTIONS:
        possibleActions.append(PropSymbolExpr(action, time = t)) # all the possible actions that pacman can take at time = t
        
    # - Pacman takes exactly one action at timestep t.
    pacphysics_sentences.append(exactlyOne(possibleActions)) # Make sure pacman takes exactly one action at timestep t
    
    # - Results of calling sensorModel(...), unless None.
    if sensorModel is not None:
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords= non_outer_wall_coords))
        
    # - Results of calling successorAxioms(...), describing how Pacman can end in various
    #         locations on this time step. Consider edge cases. Don't call if None.
    if t > 0 and successorAxioms is not None: # t > 0 because we already know where pacman is at t = 0
        pacphysics_sentences.append(successorAxioms(t, walls_grid, non_outer_wall_coords))
        
    return conjoin(pacphysics_sentences) # conjoin : AND, return all the sentences
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

    return conjoin(pacphysics_sentences)

def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent)) # 所有 wall 的位置

    "*** BEGIN YOUR CODE HERE ***"
    # 所有 t = 0 時可能的位置(sentences)
    KB.append(pacphysicsAxioms(0, all_coords, non_outer_wall_coords, walls_grid, sensorModel= None, successorAxioms=allLegalSuccessorAxioms))
    # 所有 t = 1 時可能的位置(sentences)
    KB.append(pacphysicsAxioms(1, all_coords, non_outer_wall_coords,walls_grid, sensorModel= None, successorAxioms= allLegalSuccessorAxioms))
    # Pacman 在 t = 0 時的位置
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))
    # Pacman 在 t = 0 時的行動
    KB.append(PropSymbolExpr(action0, time = 0))
    # Pacman 在 t = 1 時的行動
    KB.append(PropSymbolExpr(action1, time=1))
    # where Pacman is at (x1, y1) at time t = 1, 找到能滿足所有條件所有座位的值
    # conjoin(KB) : 所有可能的位置（KB）， PropSymbolExpr(pacman_str, x1, y1, time=1)]) : Pacman 在 (x1, y1) 上
    model1 = findModel(conjoin([conjoin(KB), PropSymbolExpr(pacman_str, x1, y1, time=1)]))
    # where Pacman is not at (x1, y1) at time t = 1, 找到能滿足所有條件所有座位的值
    # conjoin(KB) : 所有可能的位置（KB）， ~PropSymbolExpr(pacman_str, x1, y1, time=1)]) : Pacman 不在 (x1, y1) 上
    model2 = findModel(conjoin([conjoin(KB), ~PropSymbolExpr(pacman_str, x1, y1, time = 1)]))
    return (model1, model2)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"
    
#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    # Give the KB the information that pacman is at (x0, y0) at time 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))
    
    for t in range(50):
        print(t)
        
        # Give the KB the possible locations that pacman can be at time t
        locations = []
        for (x,y) in non_wall_coords:
            locations.append(PropSymbolExpr(pacman_str, x, y, time = t)) # all the possible locations that pacman can be at time t
        KB.append(exactlyOne(locations)) # Make sure pacman is at exactly one of the locations at timestep t
        
        
        act = []
        for action in actions:
            act.append(PropSymbolExpr(action, time =  t))
        KB.append(exactlyOne(act))
        
        # Try to find a model where pacman is at the goal at time t, if we find it, we're done
        model = findModel(conjoin([PropSymbolExpr(pacman_str, xg, yg, time = t), conjoin(KB)]))
        if model:
            return extractActionSequence(model, actions)
        
        # if findmodel returns false, then we need to add more information to the KB
        # Give the KB the possible locations that pacman can be at time t+1
        # (the function of pacmanSuccessorAxiomSingle is to return a sentence that describes the possible locations that pacman can be at time t+1)
        for (x,y) in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, time = t+1, walls_grid = walls_grid))
            
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"
    
#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    # Give the KB the information that pacman is at (x0, y0) at time 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))
    #initialize Food[x,y]_t variables for t = 0
    for (x,y) in food:
        KB.append(PropSymbolExpr(food_str,x,y,time = 0))
    
    for t in range(50):
        print(t)

        # Give the KB the possible locations that food will be at time t
        foodlist = []   #food variables at time = t
        for (x,y) in food:
            foodlist.append(PropSymbolExpr(food_str, x, y, time = t))

        # Give the KB the possible locations that pacman can be at time t
        locations = []
        for (x,y) in non_wall_coords:
            locations.append(PropSymbolExpr(pacman_str, x, y, time = t))
        KB.append(exactlyOne(locations))
        
        # if findmodel returns false, then we need to add more information to the KB
        act = []
        for action in actions:
            act.append(PropSymbolExpr(action, time =  t))
        KB.append(exactlyOne(act))

        for (x,y) in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, time = t+1, walls_grid = walls))

        #food successor axiom, a specific location (x,y) doesn't have food at time = t+1 if and only if pacman was there at time = t or if it didn't have food at time = t
        for (x,y) in all_coords:
            pacman = PropSymbolExpr(pacman_str,x,y,time = t)
            oldfood = PropSymbolExpr(food_str,x,y,time = t)
            newfood = PropSymbolExpr(food_str,x,y,time=t+1)
            KB.append(( pacman | ~oldfood) % ~newfood)

        goal = ~disjoin(foodlist) #all the food sentences are false (there is no food left)
        model = findModel(conjoin([goal, conjoin(KB)]))
        if model:
            return extractActionSequence(model, actions)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
