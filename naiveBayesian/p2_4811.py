"""
   Author: Nicholas Heim
   Email: nwh8@zips.uakron.edu
   Purpose: Learn naive Batesian classifiers
"""

import pandas as pd
import numpy as np

def menu():
   pass

def learn():
   pass

def save():
   pass

def newCase():
   #Need sub options here:
   pass

def quit():
   exit()


def py_nb():
   # Standard input check loop
   option = 0
   while(not(option in [1, 2, 3, 4, 5])):
         option = input(("Please choose and option and enter the number:\n"
            "1. Learn a Naïve Bayesian classifier from categorical data\n."
            "2. Save a model\n."
            "3. Load a model and testits accuracy\n."
            "4. Apply a naïve Bayesian classifier to new casesinteractively\n."
            "5. Exit the program.\n"))
   
   