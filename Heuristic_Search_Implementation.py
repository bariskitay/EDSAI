#!/usr/bin/env python
# coding: utf-8

# ## Exercise 4: Heuristic Search Implementation (5 Points)

# In this exercise, you will implement Greedy Best-First Search (GBFS). The problem we are dealing with is again Sokoban.

# Your first task is to implement the heuristic functions $h_1$, $h_2$ and $h_3$ from the exercise sheet:
# 
# - $h_1(s) = \min\limits_{c \in O(s)} MD(C(s), c)$
# - $h_2(s) = \max\limits_{c \in O(s)} MD(C(s), c)$
# - $h_3(s) = \sum\limits_{c \in O(s)} MD(C(s), c)$
# 
# Implement <code>manhattan_distance(a, b)</code> first, where <code>a</code> and <code>b</code> are 2D-points. The x and y coordinates of a point <code>p</code> are accessed by <code>p.x</code> and <code>p.y</code>.
# 
# Afterwards, use the Manhattan distance implementation to implement the heuristics. Given a state s, the data structure lets you access the character position $C(s)$ with <code>s.get_character_position()</code> and the list of positions of open crates $O(s)$ with <code>self.get_open_crate_positions()</code>. **(2 Points)**

# In[ ]:


# Manhattan distance between points a and b
def manhattan_distance(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def h1(state):
    char_pos = state.get_character_position()
    MD = []
    for i in state.get_open_crate_positions():
        MD.append(manhattan_distance(char_pos,i))
    
    if len(MD) != 0 : 
        return min(MD)
    else :
        return 0
    
def h2(state):
    char_pos = state.get_character_position()
    MD = []
    for i in state.get_open_crate_positions():
        MD.append(manhattan_distance(char_pos,i))
    
    if len(MD) != 0 :
        return max(MD)
    else :
        return 0

def h3(state):
    char_pos = state.get_character_position()
    MD = []
    for i in state.get_open_crate_positions():
        MD.append(manhattan_distance(char_pos,i))
        
    if len(MD) != 0 :
        return sum(MD)
    else :
        return 0


# Next, we will implement the actual GBFS algorithm (slide 32 in the lecture slides). As input, we will get a data structure representing our problem instance and the heuristic to use, which is just a python function. You use the needed data structures in the following way:
# 
# The problem data structure:
# - The goal test for state s corresponds to <code>problem.goal_test(s)</code>
# - The list of (applicable) actions for a state s is obtained by <code>problem.get_applicable_actions(s)</code>
# - The successor state of s when applying action a can be obtained by <code>succ = problem.generate_successor(s, a)</code>
# 
# The search node data structure:
# - To create the child node of search node n with was generated by action a and represents the state s: <code>n_child = SearchNode(s, n, a)</code>
# - The solution can be extracted from a search node by <code>n.extract_plan()</code>  (*Failure corresponds to* <code>return None</code>)
# 
# A priority queue is also already implemented. The h-ordering is already handled internally. You only need to know:
# - To insert a search node n with heuristic value h for its state: <code>frontier.insert(n, h)</code>
# - To get and remove the search node with lowest h value: <code>frontier.pop()</code>
# 
# All you have to do is put the pieces together. Complete the skeleton below. **(3 Points)**

# In[ ]:


# Load the pre-implemented search internals
from search_internals import *


# In[ ]:


def gbfs(problem, h):
    # The initial state of the problem.
    initial_state = problem.initial_state

    # The frontier ordered by ascending h.
    # Initially contains the root node.
    # Successors are added to the frontier.
    frontier = PriorityQueue()
    frontier.insert(SearchNode(initial_state), h(initial_state))
    
    # A list for duplicate checking.
    # Any state already contained in this list when generated must be skipped.
    duplicate_list = [initial_state]
    
    while True:
        if frontier.empty():
            return None
        # TODO: Empty frontier -> return failure (None)
        
        n = frontier.pop()
        # TODO: Pop the next search node
        
        if problem.goal_test(n.state):
            return n.extract_plan()
        # TODO: Goal Test
        
        for action in problem.get_applicable_actions(n.state):
            successor_state = problem.generate_successor(n.state,action)
            
            if successor_state in duplicate_list :
                continue
            
            n_prime = SearchNode(successor_state, n, action)
            
            frontier.insert(n_prime, n_prime.g + h(successor_state))
            duplicate_list.append(successor_state)
        # TODO: Generate successors
        
        # Remember to update the duplicate list!


# We provide some unit tests for your implementation, which are based on very simple Sokoban levels. You can even experiment with your own levels by designing your own in a txt file:
# 
# - \# marks a wall
# - F marks a free space
# - X marks the players position
# - C marks a crate
# - G marks a goal spot
# - B marks a goal spot that already has a crate on it
# 
# Other characters (including trailing whitespace) *should* provoke an exception. An example level could look like this:
# 
#     #####
#     #XFF#
#     #FCF#
#     #FFG#
#     #####
# 
# Be careful: While the implementation tries to check that the provided level is valid, we do not guarantee that all nonsense is rejected :-). Remember that a level is only legal if it has as many goal positions as it has crates, and a unique character position. The implementation is very simple and does not terminate for large levels (at least not with the heuristics above). To load a level: <code>problem = load_sokoban_from_file("my_level.txt")</code>. Note that we always obtain a valid solution, but depending on the heuristic we use this solution may be more complicated than needed.

# In[ ]:


problem_3 = load_sokoban_from_file("level_3.txt")
print(gbfs(problem_3, h1))

