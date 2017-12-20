import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import logit
import patsy

print "MF reloaded 1"

def score_log_1(x):
  return np.log10(x.clip(0) + 1)

def score_log(x):
  return np.log10(x)

def score_k(x,k):
  return x == k 

def score_less_k(x,k):
  return x < k

def score_more_k(x,k):
  return x >= k

def score_ref_k(x,k):
  return x - k

def score_inv(x):
  return 1./x

def mv_func(*x, **kwargs):
  """MC - Multi category encoding
  """
  #raise Exception("Not Implemented")
  levels, reference = kwargs.get("levels", None), kwargs.get("reference", None)
  weights = kwargs.get("weights", None)
  nx = len(x)
  if nx < 2:
    raise Exception("Need at least 2 columns to do multival. Otherwise just use C(column_name)")
  # If number of columns are 2 allow single weight
  # Else for each column there should be a weight
  if weights is not None:
    if len(weights.shape) == 1:
      if nx == 2:
        print "using complimentary weights for 2 columns. w and 1-w"
        weights = np.insert(weights[:, np.newaxis], 1, 1. - weights, axis=1) # Add complimentary weights
      else:
        weights = weights[:, np.newaxis] # Create weights into 1 column matrix
    elif nx > 1 and weights.shape[1] != nx:
      raise Exception("Either weights should be a 1d array or 2d array with number of columns equal to %s" % nx)
  else:
    weights = np.ones((x[0].shape[0], 1)) # Create 1 column all ones weight matrix
  if len(x[0].shape) != 1:
    raise Exception("Mismatching Shapes. All arrays should be 1d and should have the same shape")
  for k in x:
    if k.shape != x[0].shape:
      raise Exception("Mismatching Shapes. All arrays should be 1d and should have the same shape")
  if levels is None:
    levels = np.sort(np.unique(np.hstack(x))) # Sort the unique values and then use this ordering as levels
  else:
    levels = np.array(levels)
  if reference is None:
    reference = levels[0]
  #print "Levels: %s, reference: %s" % (levels, reference)
  levels = levels[levels != reference] # Remove reference from levels
  level_len = len(levels)
  #print x[0].shape[0], level_len
  out = np.zeros((x[0].shape[0], level_len))
  for i, v in enumerate(levels):
    # print i, v
    for j,col in enumerate(x):
      idx = np.where(np.array(col) == v)
      out[idx, i] = weights[idx, min(nx-1,j)]
  #print "Created matrix with shape: ", out.shape
  colnames = ["T.%s" % k for k in levels]
  return pd.DataFrame(out, columns=colnames)
  # Use patsy contrast matrix
  # contrast = Treatment(reference='N').code_without_intercept(levels) # Levels should be a list
  # contrast.matrix[x, :] Here x should be array of index of categories in levels

class MultiVal(object):
  def __init__(self):
    print "Using class based MultiVal"
    self.levels = []
    self.reference = []
    self.colnames = []
    self.level_len = 0
    self.nx = 0
    self.levels_given = False
    self.verbose = False

  def memorize_chunk(self,*x,**kwargs):
    self.verbose = kwargs.get("verbose", False)
    levels, reference = kwargs.get("levels", None), kwargs.get("reference", None)
    self.nx = len(x)
    if self.nx < 2:
      raise Exception("Need at least 2 columns to do multival. Otherwise just use C(column_name)")
    # If number of columns are 2 allow single weight
    # Else for each column there should be a weight
    if len(x[0].shape) != 1:
      raise Exception("Mismatching Shapes. All arrays should be 1d and should have the same shape")
    for k in x:
      if k.shape != x[0].shape:
        raise Exception("Mismatching Shapes. All arrays should be 1d and should have the same shape")
    if levels is None:
      self.levels.extend(np.sort(np.unique(np.hstack(x))).tolist())
    else:
      self.levels_given = True
      self.levels = levels
    self.reference = reference
    if self.verbose:
      print "In memorize chunk"
      print "Levels given: %s" % self.levels_given
      print "LEVELS: %s" % levels
      print "REFERENCE: %s" % reference
      print "self.LEVELS: %s" % self.levels
      print "self.REFERENCE: %s" % self.reference

  def memorize_finish(self):
    if not self.levels_given:
      self.levels = np.array(list(set(self.levels)))
    else:
      self.levels = np.array(self.levels)
    if self.reference is None:
      self.reference = self.levels[0]
    self.levels = self.levels[self.levels != self.reference] # Remove reference from levels
    self.level_len = len(self.levels)
    self.colnames = ["T.%s" % k for k in self.levels]
    if self.verbose:
      print "In memorize finish"
      print "Levels given: %s" % self.levels_given
      print "self.LEVELS[%s]: %s" % (self.level_len, self.levels)
      print "self.REFERENCE: %s" % self.reference

  def transform(self, *x, **kwargs):
    out = np.zeros((x[0].shape[0], self.level_len))
    weights = kwargs.get("weights", None)
    if weights is None:
      weights = np.ones((x[0].shape[0], 1)) # Create 1 column all ones weight matrix
    if len(weights.shape) == 1:
      if self.nx == 2:
        print "using complimentary weights for 2 columns. w and 1-w"
        weights = np.insert(weights[:, np.newaxis], 1, 1. - weights, axis=1) # Add complimentary weights
      else:
        weights = weights[:, np.newaxis] # Create weights into 1 column matrix
    elif self.nx > 1 and weights.shape[1] != self.nx and weights.shape[1] != 1:
      raise Exception("Either weights should be a 1d array or 2d array with number of columns equal to %s" % self.nx)
    if self.verbose:
      print "In transform"
      print "x: %s, self.nx: %s" % (len(x), self.nx)
      print "out.shape: %s, weights.shape: %s" % (out.shape, weights.shape)
      print "Levels given: %s" % self.levels_given
      print "self.LEVELS[%s]: %s" % (self.level_len, self.levels)
      print "self.REFERENCE: %s" % self.reference
    for i, v in enumerate(self.levels):
      for j,col in enumerate(x):
        col = col.values
        idx = np.where(np.array(col) == v)[0]
        w_col = min(weights.shape[1] -1,j)
        if self.verbose:
          print "Level: %s, %s" % (i,v)
          print "Column: %s, W_col: %s" % (j,w_col)
          print "Found values: %s" % (idx.shape,)
          print "Setting values: %s" % (out[idx, i].shape,)
          print "Setting weights: %s" % (weights[idx, w_col].shape,)
          print "Unique weights: %s" % (np.unique(weights[idx, w_col]))
        # Add the weights so as to allow for cases UNKNOWN, UNKNOWN
        out[idx, i] += weights[idx, w_col] 
    return pd.DataFrame(out, columns=self.colnames, index=x[0].index)

MC = patsy.stateful_transform(MultiVal)
