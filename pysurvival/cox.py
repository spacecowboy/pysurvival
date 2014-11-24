# -*- coding: utf-8 -*-
"""
This script provides the code necessary to create and evaluate a cox model.
Code from R is called and the library 'survival' is required to be installed.

@author: Jonas Kalderstam
"""
from __future__ import division
from rpy2.robjects.packages import importr
from rpy2.robjects import r as r
from rpy2 import robjects

# To convert Numpy to R-vectors,
# we need to import this and activate said conversion
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
# rpy2.robjects.activate()
# To create R vectors
from rpy2.robjects.vectors import FloatVector
# For cox modeling
__survival = importr('survival')

import numpy as np

class committee:
    '''
    An ensamble of Cox Models. Used for cross validation. Array_of_data is a list of data sets,
    where each dataset is a tuple as (training, validation)
    '''
    def __len__(self):
        return len(self.members)

    def __init__(self, array_of_data, targetcol, eventcol, headers=None):
        self.members = []
        for data in array_of_data:
            cox = cox_model(data[0], targetcol, eventcol, headers)
            cox.internal_set = (data) #Includes validation set
            self.members.append(cox)


    def risk_eval(self, input_array, cox = None):
        """
        Returns the average index value for input over all members of the committee, unless cox is specified
        it will then perform a risk_eval for that cox model only (doesn't have to be a member even)
        DO NOT INCLUDE TARGET/EVENTS COLUMNS AS INPUT HERE!
        """

        if cox is None:
            avg_index = 0.0
            for cox in self.members:
                output = cox.hazard_eval(input_array)
                #Greater than since cox will estimate the hazard, not survival time
                index = len(cox.trn_set[cox.trn_set > output]) # Length of the array of all values greater than output = index where output would be placed
                #Normalize it
                avg_index += index / float((len(cox.trn_set) + 1)) # +1 to make sure the maximum is 1.0 if the input is placed last

            return avg_index / float(len(self))
        else:
            output = cox.hazard_eval(input_array)
            #Greater than since cox will estimate the hazard, not survival time
            index = len(cox.trn_set[cox.trn_set > output]) # Length of the array of all values less than output = index where output would be placed
            #Normalize it
            return index / (len(cox.trn_set) + 1)

class cox_model:
    def __init__(self, data, targetcol, eventcol, headers=None):
        self.colcount = data.shape[1]
        self.data = data
        self.coxfit = _cox_proportional_hazard(self.data, targetcol,
                                               eventcol, headers)
        self.targetcol = targetcol
        self.eventcol = eventcol
        self.headers = headers
        self.trn_set = _cox_predict(self.coxfit)

    def risk_eval(self, input_array):
        '''
        Convenience function. The minus makes the output have the same
        rank order as models which output survival prediction. Useful
        for calculating the concordance index.
        '''
        return -self.hazard_eval(input_array)

    def hazard_eval(self, input_array):
        if self.coxfit is None:
            raise ValueError('No model has been trained!')
        elif input_array.ndim != 2:
            raise ValueError("Input is expected to be 2-dimensional!")
        elif self.colcount != input_array.shape[1]:
            raise ValueError("Input has an incorrect number of columns: " +
                             "{}, expected {}".format(input_array.shape[1],
                                                      self.colcount))

        #Dimensions must match in R
        return _cox_predict(self.coxfit, input_array, self.headers)

    def hazard_eval_training(self):
        if self.coxfit is None:
            raise ValueError('No model has been trained!')
        else:
            return _cox_predict(self.coxfit)


def _convert_to_dataframe(array, headers = None):
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


def _cox_proportional_hazard(data, targetcol, eventcol, headers=None):
    '''
    This function will create a Cox model for the input data given and
    return that model.
    Input is expected to be a 2-dimensional numpy array (or pandas dataframe).
    target_col and event_col are the names of the columns. If headers are not
    used, then these must be strings of the columns as 'X0', 'X1', etc. These
    are the default column names in R. If a pandas dataframe is specified,
    then the column names of the dataframe are used (unless headers are
    specified separately).
    '''
    # First convert the input data from numpy to an R dataframe
    r_data = _convert_to_dataframe(data, headers)
    names = list(r_data.names)
    # Remove target variables
    names.remove(targetcol)
    names.remove(eventcol)

    # Next we want to create a Cox model for this data
    # We need to insert the variables into the R environment for ease of use
    robjects.globalenv['r_data'] = r_data
    # coxfit = coxph(Surv(target_col, event_col) ~ inputnames, r_data)
    cmd = 'coxfit <- coxph(Surv({0}, {1}) ~ {2}'.format(targetcol, eventcol,
                                                        names[0])
    # Add remaining input columns
    for name in names[1:]:
        cmd += '+{0}'.format(name)
    cmd += ', r_data, model=TRUE)'
    coxfit = r(cmd)
    # Print summary of model
    #print(r('summary(coxfit)'))

    return coxfit


def _cox_predict(coxfit, npdata=None, headers=None):
    '''
    Returns an output array, of the same length as the input array.
    Each output in this array is the prediction of the
    cox model and represents the relative risk for all the patients.
    Input is expected to be a numpy array of dimension 2 where each patient
    is represented by a row and each covariate
    by a column. The headers and format of the data must match that of the
    training data!
    '''
    # insert into environment
    robjects.globalenv['coxfit'] = coxfit
    # Use linear(lp) for risk prediction
    if npdata is None:
        prediction = r('predict(coxfit, type="lp")')
    else:
        # First convert the input data from numpy to an R dataframe
        r_data = _convert_to_dataframe(npdata, headers)
        robjects.globalenv['pred_data'] = r_data
        # We want to use the exponent directly and bypass the baseline
        prediction = r('predict(coxfit, newdata=pred_data, type="lp")')

    # Convert this back to a numpy array and return!
    result = np.array(prediction)
    return result


if __name__ == '__main__':
    #Make sure it works with both names and not
    fit = _cox_proportional_hazard(np.array([[2.0, 1.0, 4.0], [4.0, 1.0, 8.0], [6.0, 1.0, 12.0], [8.0, 1.0, 16.0],
                                  [10.0, 1.0, 20.0], [12.0, 1.0, 24.0], [14.0, 0.0, 16.0]]), 'X2', 'X1')

    fit = _cox_proportional_hazard(np.array([[2.0, 1.0, 4.0], [4.0, 1.0, 8.0], [6.0, 1.0, 12.0], [8.0, 1.0, 16.0],
                                  [10.0, 1.0, 20.0], [12.0, 1.0, 24.0], [14.0, 0.0, 16.0]]), 'c', 'b', ['a', 'b', 'c'])

    print(_cox_predict(fit))
    print(_cox_predict(fit, np.array([[20.0, 0.0, 4.0], [15.0, 0.0, 8.0], [10.0, 0.0, 12.0], [5.0, 0.0, 5.0]]),
                      ['a', 'b', 'c']))

    data_set = []
    data_set.append((np.array([[2.0, 1.0, 4.0], [4.0, 1.0, 8.0], [6.0, 1.0, 12.0], [8.0, 1.0, 16.0],
                                  [10.0, 1.0, 20.0], [12.0, 1.0, 24.0], [14.0, 0.0, 16.0]]), None))
    data_set.append((np.array([[3.0, 1.0, 6.0], [ 5.0, 1.0, 10.0], [7.0, 1.0, 14.0], [9.0, 1.0, 18.0],
                                  [11.0, 1.0, 22.0], [13.0, 1.0, 26.0], [14.0, 0.0, 16.0]]), None))

    com = committee(data_set, 'c', 'b', ['a', 'b', 'c'])
    print("Committee results")
    print(com.risk_eval(np.array([4])))
    print(com.risk_eval(np.array([2]), cox = com.members[0]))
    print("Individually\nInput, Index (Survival Chance)")
    for _d in data_set:
        for _p in _d[0]:
            print(_p[0], com.risk_eval(np.array([_p[0]])))
    print("Member hazard")
    print(1.0, com.members[0].hazard_eval(np.array([1.0])))
