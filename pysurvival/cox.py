# -*- coding: utf-8 -*-
"""
This script provides the code necessary to create and evaluate a cox model.
Code from R is called and the library 'survival' is required to be installed.

@author: Jonas Kalderstam
"""
from __future__ import division
# Needed to fix issue with rpy2 and Conda
import readline
from rpy2.robjects.packages import importr
from rpy2.robjects import r as r
from rpy2 import robjects

# To convert Numpy to R-vectors,
# we need to import this and activate said conversion
#from rpy2.robjects.numpy2ri import numpy2ri
#robjects.conversion.py2ri = numpy2ri
#robjects.activate()
# To create R vectors
from rpy2.robjects.vectors import FloatVector
# For cox modeling
__survival = importr('survival')

import numpy as np
import pandas as pd


class CoxModel(object):
    def __init__(self):
        self.xcols = None
        self.coxfit = None

    def fit(self, df, duration_col, event_col):
        '''
        Fit a data frame.
        '''
        self.xcols = df.columns - [duration_col, event_col]
        r_df = _convert_to_dataframe(df)
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df

        # Build command
        cmd = ('coxfit = coxph(Surv(r_df${time}, r_df${event}) ~ {incols}, '
               + 'data=r_df, model=TRUE)').format(time=duration_col,
                                                  event=event_col,
                                                  incols='+'.join(self.xcols))
        self.coxfit = r(cmd)

    def risk_eval(self, df):
        '''
        Convenience function. The minus makes the output have the same
        rank order as models which output survival prediction. Useful
        for calculating the concordance index.
        '''
        return -self.hazard_eval(df)

    def summary(self):
        '''
        Print R's summary
        '''
        if self.coxfit is None:
            raise ValueError('No model has been trained!')
        print(r('summary(coxfit)'))

    def predict(self, df):
        return self.hazard_eval(df)

    def hazard_eval(self, df):
        if self.xcols is None or self.coxfit is None:
            raise ValueError("No model has been trained yet!")

        r_df = _convert_to_dataframe(df)
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df

        # Make the command
        prediction = r('predict(coxfit, newdata=r_df, type="lp")')

        # Return np array of it
        return np.array(prediction)


def _convert_to_dataframe(array, headers=None):
    '''
    Convert from a numpy array of dimension 2 to an R data frame.
    np_list should be a 2 dimensional array of floats and headers, if desired,
    is a list of strings corresponding to the number of columns in np_list
    (also matching their indices).
    If no headers are specified, then the number of the columns are used as
    names, as is the default in R.
    '''
    # array could be a dataframe. Use its headers if possible
    if headers is None:
        try:
            headers = array.columns
            # Need to be an array for iteration below
            array = np.array(array)
        except AttributeError:
            # It's not a dataframe
            headers = ['X{}'.format(i) for i in range(array.shape[1])]

    #First, we create a dictionary of the columns
    d = {}

    for col, name in enumerate(headers):
        d[name] = FloatVector(array[:, col])
    #Now, we can create a data frame from these R-vectors
    data = robjects.DataFrame(d)
    return data
