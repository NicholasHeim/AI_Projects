
def writeStep(state, file):
   for i in [0,1,2]:
      file.write('%-1s' % state[3*i:3*(i+1)] + '\n')
   file.write('\n')

def n_puzzle():
   # Get the initial inputs for the program and open the file. This includes the 
   # initial and final states of the board. This should be the only required input.
   start = list(map(int, input("\nEnter the initial state: ").strip().split()))
   end = list(map(int, input("\nEnter the final state: ").strip().split()))
   result = open("8puzzlelog.txt", mode='w')
   writeStep(start, result)






   result.close()
