"""
   Author: Nicholas Heim
   Email: nwh8@zips.uakron.edu
   Purpose: Learn naive Batesian classifiers
"""

import pandas as pd
import numpy as np

"""
   NOTE: I do not claim authorship of lines 15-54. These are attributed to Dr. Chan
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
    d_prior = freq(df[decision_list], 'dict')
    c_list = inverse_p(df, cond_list, decision_list)
    return (d_prior, c_list, cond_list, decision_list)

# Helper function for outputting the menu to the user.
def menu():
   return input(("Please choose and option and enter the number:\n"
      "1. Learn a Naïve Bayesian classifier from categorical data\n."
      "2. Save a model\n."
      "3. Load a model and testits accuracy\n."
      "4. Apply a naïve Bayesian classifier to new casesinteractively\n."
      "5. Exit the program.\n"))

def learn():
   fileName = input(("Enter the name of the data file excluding the extension:\n"
      "Note: The extension is assumed to be .csv\n"))
   df = pd.read_csv(fileName + ".csv")
   d_prior, c_list, cond_list, decision_list = bayes_model(df)

   print(("Naive Bayesian model learned.\n"))

   return (cond_list, decision_list, d_prior, c_list, fileName)

def save(fileName):
   file = open(fileName + ".bin", "wb")


   
   file.close()

def load():
   fileName = input(("Enter the name of the model file excluding the extension:\n"
      "Note: The extension is assumed to be .bin\n"))
   file = open(fileName + ".bin", "rb")

   df = pd.read_csv(input(("Enter the name of the data file excluding the extension:\n"
      "Note: The extension is assumed to be .csv\n")))
   
   

   pass

def newCase():
   option = int(input(("Please choose and option and enter the number:\n"
                   "1. Enter a new case interactively.\n"
                   "2. Return to the first menu.\n")))
   while(option != 2):
      while(not(option in ['1', '2'])):
         option = input(("Please choose and option and enter the number:\n"
                         "1. Enter a new case interactively.\n"
                         "2. Return to first menu.\n"))

      if option == 1:
         pass
   

def py_nb():
   while True:
      option = int(menu())
      # Standard input check loop
      while(not(option in [1, 2, 3, 4, 5])):
         print("The value entered was not a valid option.\n")
         option = int(menu())

      if option == 1:
         cond_list, decision_list, d_prior, c_list, fileName = learn()
      elif option == 2 and fileName != '':
         save(fileName)
      elif option == 3:
         load()
      elif option == 4:
         newCase()
      else:
         raise SystemExit


# Calling the main driver function on run
py_nb()