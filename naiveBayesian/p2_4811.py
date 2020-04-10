"""
   Author: Nicholas Heim
   Email: nwh8@zips.uakron.edu
   Purpose: Learn naive Batesian classifiers
"""

# TODO: Comment 

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import array
import io

"""
   NOTE: I do not claim authorship of lines 19-58. These are attributed to Dr. Chan
   At the University of Akron. All else I claim.
"""

#represent frequency count of one feature as a DataFrame
def freq(x, opt='DataFrame'):
    """ x is a Series
        it returns a DataFrame (by default) indexed by unique values of x and
        their frequency counts
    """
    if opt != 'DataFrame':
        if opt == 'dict':
            return { i: x.value_counts()[i] for i in x.unique()}
        else:
            return (x.name, { i: x.value_counts()[i] for i in x.unique()})
    return pd.DataFrame([x.value_counts()[i] for i in x.unique()], index=x.unique(), columns=[x.name])

#How to create multi-index objects?
#Use  groupby()
def cond_p(df, c, d):
    """ compute p(d|c)
        represented as a dict
        df is a DataFrame with columns c and d
        c and d are column names
    """
    C = df.groupby(c).groups
    D = df.groupby(d).groups
    P_DC = { (i, j): (C[i] & D[j]).size / C[i].size
                 for i in C.keys() for j in D.keys()}
    
    return P_DC  #returns P(d|c) as a dict

def inverse_p(df, cond_list, decision_list):
    """ Build a list of dict of inverse probabilities
    """
    p_list = [cond_p(df, decision_list, i) for i in cond_list] #build a list of dicts
    return p_list

def bayes_model(df):
    cond_list = df.columns[:-1]  #get the list of condition attributes
    decision_list = df.columns[-1]  #get the decision attribute, assumed to be the last one
    #d_prior = freq(df[decision_list], 'dict')
    c_list = inverse_p(df, cond_list, decision_list)
    return (c_list)

# Helper function for outputting the menu to the user.
def menu():
   return input(("\nPlease choose an option and enter only the number:\n"
      "1. Learn a Naïve Bayesian classifier from categorical data.\n"
      "2. Save a model.\n"
      "3. Load a model and test its accuracy.\n"
      "4. Apply a naïve Bayesian classifier to new cases interactively.\n"
      "5. Exit the program.\n"
      "Please choose either 1 or 3 first, as there is no error checking in 2 and 4 currently.\n"))

class Node:
   # position will give the position in options the next level was taken from
   def __init__(self, splitWidth, options, probabilities, position):
      # Left will the first one, right will be the last, as you would read it.
      self.linked = []
      if(position >= 0):
         # Set values for this node:
         # Left and right refer to the decision values, left being first, right second
         self.leftProb = float(probabilities[0][position])
         self.rightProb = float(probabilities[0][int(position + (len(probabilities[0]) / 2))])


         # Check to see if this is a leaf node:
         # Note, should not ever actually be < 1
         if len(options)  >= 2:
            # Set values for next nodes:
            self.nexts = options[1]
            # Holds the options for the next row of the tree
            # Create the next level of nodes
            for i in range(splitWidth[1]):
               self.linked.append(Node(splitWidth[1:], options[1:], probabilities[1:], i))
         else:
            self.nexts = []

      # Type == -1 means that it is the head node of the tree
      else:
         self.nexts = options[0]
         for i in range(splitWidth[0]):
            self.linked.append(Node(splitWidth, options, probabilities, i))
   
def test(head, data):
   # The two probabilities that will be compared in the end
   left = 1
   right = 1
   linked = len(head.nexts)
   current = head
   while(linked != 0):
      for i in range(len(current.nexts)):
         if data[0] == current.nexts[i]:
            pos = i
            break
         
      current = current.linked[pos]
      left = left * current.leftProb
      right = right * current.rightProb

      linked = len(current.linked)
      if linked != 0:
         data = data[1:]

   return left, right

def constructTree(model):
   # Seperate the model data into the rows it will be in by legal {} placements
   # model will be a string of the model data
   model = str(model)
   data = []
   splitWidth = []
   options = []
   probabilities = []
   for i in range(model.count('{')):
      start = model.find('{')
      end = model.find('}')
      data.append(model[start + 1:end + 1])
      model = model[end + 1:]

      # First string in parenthesis is the decision variable, ASSUMED BINARY
      splitWidth.append(data[i].count(data[i][1 : data[i].find("'", 2) + 1]))
      
      # Gather options for each level of the tree
      # trashData is used to avoid losing the last characters in the sliced string
      temp = []
      trashData = data[i][:]
      for j in range(splitWidth[i]):
         temp.append(trashData[trashData.find(',') + 2 : trashData.find(')')])
         temp[j] = temp[j].replace("'", "")
         trashData = trashData[trashData.find(',', trashData.find(':')) + 2 :]
      
      options.append(temp)
      
      # Pull out the probabilities for each value.
      temp = []
      trashData = data[i][:]
      for j in range((2 * splitWidth[i]) - 1):
         temp.append(float(trashData[trashData.find(':') + 1 : trashData.find(',', trashData.find(':'))]))
         trashData = trashData[trashData.find(',', trashData.find(':')) + 2 :]

      temp.append(float(data[i][data[i].rfind(':') + 1 : data[i].find('}')]))
      probabilities.append(temp)
   
   # Adding the decision values into the last spot in options
   temp = []
   temp.append(data[0][2 : data[0].find("'", 2)])
   temp.append(data[0][data[0].rfind("(") + 2 : data[0].rfind(',') - 1])
   options.append(temp)

   # At this point, the data is ready to be fed into the tree.
   # Need splitWidth, options, probabilities, position = -1
   return (Node(splitWidth, options[:-1], probabilities, -1), options[-1:][0])

def learn():
   fileName = input(("\nEnter the name of the data file excluding the extension:\n"
      "Note: The extension is assumed to be .csv\n"))
   c_list = bayes_model(pd.read_csv(fileName + ".csv"))

   # Generate the model tree
   head, decisions = constructTree(c_list)
   print(("Naive Bayesian model learned.\n"))
   return (c_list, fileName, head, decisions)

def save(fileName, model):
   file = open(fileName + ".bin", "wb")
   out = bytearray(str(model), 'ascii')
   file.write(out)
   file.close()

def load():
   fileName = input(("\nEnter the name of the model file excluding the extension:\n"
      "Note: The extension is assumed to be .bin\n"))
   file = open(fileName + ".bin", "rb")

   model = file.read()
   model = model.decode('ascii')

   head, decisions = constructTree(model)

   df = pd.read_csv(input(("\nEnter the name of the data file for testing, exclude the extension:\n"
      "Note: The extension is assumed to be .csv\n")) + '.csv')
   data = []
   actual = []
   data.append(list(df))
   actual.append(data[0][-1:][0])
   df = df.values.tolist()
   for i in range(len(df)):
      data.append(df[i])
      actual.append(data[i + 1][-1:][0])
      data[i] = data[i][:-1]
   
   predicted = []
   for i in range(len(data)):
      left, right = test(head, data[i])
      if(left > right):
         predicted.append(decisions[0])
      elif (right > left): 
         predicted.append(decisions[1])
   
   print(decisions)
   print(predicted)
   file.close()

   # Construction of the confusion matrix
   df = {'Actual':      actual,
         'Predicted':   predicted   }

   df = pd.DataFrame(df, columns=['Actual', 'Predicted'])
   df['Actual'] = df['Actual'].map({decisions[0]: 0, decisions[1]: 1})
   df['Predicted'] = df['Predicted'].map({decisions[0]: 0, decisions[1]: 1})

   cMatrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], 
                         colnames=['Predicted'], normalize=True)
   sn.heatmap(cMatrix, annot=True)
   plt.show()

   return (head, decisions, '')
   
def newCase(head, decisions):
   option = int(input(("\nPlease choose an option and enter only the number:\n"
                   "1. Enter a new case interactively.\n"
                   "2. Return to the first menu.\n")))
   while(not(option in [1, 2])):
      option = int(input(("\nPlease choose an option and enter only the number:\n"
                   "1. Enter a new case interactively.\n"
                   "2. Return to first menu.\n")))

   while True:
      if option == 2:
         return
      
      case = list(map(str, input(("\nEnter the case with each condition attribute "
                        "ONLY separated by a single space.\n"
                        "NOTE: capitalization matters.\n")).strip().split()))
      left, right = test(head, case)
      if(left > right):
         print(str(left) + ' > ' + str(right) + '\nPrediction: ' + decisions[0])
      else:
         print(str(left) + ' < ' + str(right) + '\nPrediction: ' + decisions[1])

      option = int(input(("\nPlease choose an option and enter only the number:\n"
                   "1. Enter a new case interactively.\n"
                   "2. Return to the first menu.\n")))
      while(not(option in [1, 2])):
         option = int(input(("\nPlease choose an option and enter only the number:\n"
                      "1. Enter a new case interactively.\n"
                      "2. Return to first menu.\n")))

def py_nb():
   while True:
      option = int(menu())
      # Standard input check loop
      while(not(option in [1, 2, 3, 4, 5])):
         print("The value entered was not a valid option.\n")
         option = int(menu())

      if option == 1:
         c_list, fileName, head, decisions = learn()
      elif option == 2 and fileName != '':
         save(fileName, c_list)
         print('Model saved.\n')
      elif option == 3:
         head, decisions, fileName = load()
      elif option == 4:
         newCase(head, decisions)
      elif option == 5:
         raise SystemExit

py_nb()