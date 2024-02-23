"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

import numpy as np

def scale(X):
    for element in X:
      if not isinstance(element, float):
        return False
      if not 0.0 <= element <= 1.0:
        return False
    return True
    
def scale_data(X):
   X = (X-X.min())/(X.max() - X.min())
   return X


