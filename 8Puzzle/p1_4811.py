"""
   Author: Nicholas Heim
   Email: nwh8@zips.uakron.edu
   Purpose: To find a sequence of moves to solve an 8 puzzle game 
            given an initial and final configuration. 
"""
import time as t

# Changing the recall as a list, need to change: 
# write_step
# State.init
# PriorityQueue.h_dequeue

def write_step(stateList, file):
   file.write("\nMoves to solution:\n\n")
   for i in stateList:
      for j in [0, 1, 2]:
         file.write('%-1s' % i[3*j:3*(j+1)] + '\n')
      file.write('\n')

def swap(list, pos1, pos2):
    temp = list[:]
    temp[pos1], temp[pos2] = temp[pos2], temp[pos1] 
    return temp

class State:
   def __init__(self, state, moves, previous):
      self.state = state
      self.moves = moves
      self.previous = []
      self.previous.extend(previous)
      self.previous.append(self.state)
      self.heur = 0
   
   # Using hamming distance + moves for the heuristic
   def calc_ham(self, end):
      sum = 0
      for i, j in zip(self.state, end):
         if (i != j):
            sum += 1
      self.heur = sum + self.moves

   def calc_bfs(self):
      self.heur = self.moves

   # Using manhattan distance + moves for the heuristic
   def calc_man(self, end):
      vals = []
      for i in self.state:
         # Do not care about the heuristic of 0, as that is the open space
         if i != 0:
            pos = self.state.index(i)
            epos = end.index(i)
            # Following if statement determines if the two values are in the same group
            # of three, allowing for an easy calculation
            if ((pos < 3 and epos < 3) or 
               (2 < pos and 2 < epos and pos < 6 and epos < 6) or 
               (5 < pos and 5 < epos)):
               vals.append(abs(pos - epos))
            # Determines if the values are two rows apart
            elif ((pos < 3 and epos > 5) or (pos > 5 and epos < 3)):
               vals.append(2 + abs((pos % 3) - (epos % 3)))
            # Assumes the values are 1 row apart, the last option
            else:
               vals.append(1 + abs((pos % 3) - (epos % 3)))
      sum = 0
      for i in vals:
         sum += i
      self.man = sum + self.moves

# Min priority queue. Meant to organize based on the value of the heuristic
class PriorityQueue:
   def __init__(self, State, end):
      self.queue = []
      self.queue.append(State)
      self.end = end
      self.done = False
   
   def is_empty(self):
      return len(self.queue) == []

   def enqueue(self, State):
      self.queue.append(State)

   def h_dequeue(self):
      min = 0
      for i in range(len(self.queue)):
         if self.queue[i].heur < self.queue[min].heur:
            min = i
      
      rem = self.queue[min]
      self.end_check(rem)
      del self.queue[min]

      for i in self.moves(rem.state, rem.previous[len(rem.previous) - 2]):
         temp = State(i, rem.moves + 1, rem.previous)
         temp.calc_ham(self.end)
         self.queue.append(temp)
      return rem

   # Note: 
   def b_dequeue(self):
      min = 0
      for i in range(len(self.queue)):
         if self.queue[i].heur < self.queue[min].heur:
            min = i
      
      rem = self.queue[min]
      self.end_check(rem)
      del self.queue[min]

      for i in self.moves(rem.state, rem.previous[len(rem.previous) - 2]):
         temp = State(i, rem.moves + 1, rem.previous)
         temp.calc_bfs()
         self.queue.append(temp)
      return rem

   def m_dequeue(self):
      min = 0
      for i in range(len(self.queue)):
         if self.queue[i].man < self.queue[min].man:
            min = i
         temp = self.queue[min]
         del self.queue[min]
         return temp

   def moves(self, state, prev):
      moves = []
      zero = state.index(0)
      # Corner cases, 2 options, 1 duplicate
      if zero in [0, 2, 6, 8]:
         if zero == 0:
            for i in [1, 3]:
               if (swap(state, 0, i)) != prev:
                  moves.append(swap(state, 0, i))
         elif zero == 2:
            for i in [1, 5]:
               if (swap(state, 2, i)) != prev:
                  moves.append(swap(state, 2, i))
         elif zero == 6:
            for i in [3, 7]:
               if (swap(state, 6, i)) != prev:
                  moves.append(swap(state, 6, i))
         elif zero == 8:
            for i in [5, 7]:
               if (swap(state, 8, i)) != prev:
                  moves.append(swap(state, 8, i))
      # Middle side cases, 3 options, 1 duplicate
      elif zero in [1, 3, 5, 7]:
         if zero == 1:
            for i in [0, 2, 4]:
               if (swap(state, 1, i)) != prev:
                  moves.append(swap(state, 1, i))
         elif zero == 3:
            for i in [0, 4, 6]:
               if (swap(state, 3, i)) != prev:
                  moves.append(swap(state, 3, i))
         elif zero == 5:
            for i in [2, 4, 8]:
               if (swap(state, 5, i)) != prev:
                  moves.append(swap(state, 5, i))
         elif zero == 7:
            for i in [4, 6, 8]:
               if (swap(state, 7, i)) != prev:
                  moves.append(swap(state, 7, i))
      # Center case, 1 option, 1 duplicate
      else:
         for i in [1, 3, 5, 7]:
            if (swap(state, 4, i)) != prev:
               moves.append(swap(state, 4, i))
      return moves
   
   def end_check(self, state):
      for i, j in zip(state.state, self.end):
         if i != j:
            return
      self.done = True

def n_puzzle():
   # Get the initial inputs for the program and open the file. This includes the 
   # initial and final states of the board. This should be the only required input.
   # It is expected that the user knows that the numbers 0-8 are used as inputs.
   start = list(map(int, input("\nEnter the initial state: ").strip().split()))
   end = list(map(int, input("\nEnter the final state: ").strip().split()))
   result = open("8puzzlelog.txt", mode='w')

   # Hard coded testing cases, last one is one with no solution.
   #start = [1, 2, 3, 4, 6, 0, 7, 5, 8]
   #start = [0, 1, 3, 4, 2, 5, 7, 8, 6]
   #start = [1, 2, 3, 4, 5, 6, 7, 0, 8]
   #start = [1, 2, 3, 4, 5, 6, 8, 7, 0]
   #end = [1, 2, 3, 4, 5, 6, 7, 8, 0]

   # Start of the brute force method using breadth first search and additional
   # functions within the priority queue. These will be used here.
   # Note: The moves will not be output until the end of the 
   # heuristic solution to allow for a more ccurate comparison of times. 
   t0 = t.time()
   pq = PriorityQueue(State(start, 0, []), end)
   removed = pq.b_dequeue()
   while (pq.done == False) and (not pq.is_empty()) and (removed.moves < 32):
      removed = pq.b_dequeue()

   result.write("Non-Heuristic Time: " + (str)(t.time() - t0) + '\n')
   result.write("Non-Heuristic Moves: " + (str)(removed.moves) + "\n\n")
   

   # Start of the heuristic solution. Metric: hamming distance. 
   t0 = t.time()
   pq = PriorityQueue(State(start, 0, []), end)
   
   removed = pq.h_dequeue()
   while (pq.done == False) and (not pq.is_empty()) and (removed.moves < 32):
      removed = pq.h_dequeue()

   result.write("Heuristic Time: " + (str)(t.time() - t0) + '\n')
   result.write("Heuristic Moves: " + (str)(removed.moves) + '\n')

   write_step(removed.previous, result)
   result.close()

n_puzzle()

# Look at corountine()
# condition variable for multithreading