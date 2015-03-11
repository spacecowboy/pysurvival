# -*- coding: utf-8 -*-
"""
Wraps the RPart model.
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
# To create R vectors
#from rpy2.robjects.vectors import FloatVector
# R imports
__rpart = importr('rpart')
__survival = importr('survival')

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from .cox import _convert_to_dataframe


class RPartModel(object):
    def __init__(self, highlim=0.25, lowlim=0.25,
                 minsplit=None, minbucket=None,
                 xval=None, cp=None):
        '''
        Parameters:
        highlim - Minimum size of high risk group. Default 0.25.
        lowlim - Minimum size of low risk group. Default 0.25.

        R-parameters (copied from rpart manual):
        minsplit - The minimum number of observations in a node for which the
                   routine will even try to compute a split. The default is 20.
                   This parameter can save computation time, since smaller nodes
                   are almost always pruned away by cross-validation.
        minbucket - The minimum number of observations in a terminal node. This
                    defaults to minsplit/3.
        xval - The number of cross-validations to be done. Usually set to zero
               during exploratory phases of the analysis. A value of 10, for
               instance, increases the compute time to 11-fold over a value
               of 0.
        cp - The threshold complexity parameter. Default value is 0.01. Should
             be a value between 0 and 1. A value of cp = 1 will always result
             in a tree with no splits. The default value of .01 has been
             reasonably successful at "pre-pruning" trees so that the
             cross-validation step need only remove 1 or 2 layers, but it
             sometimes over prunes, particularly for large data sets.
        '''
        self.highlim = highlim
        self.lowlim = lowlim
        self.minbucket = minbucket
        self.minsplit = minsplit
        self.xval = xval
        self.cp = cp
        self.xcols = None
        self.myfit = None

    def fit(self, df, duration_col, event_col):
        '''
        Fit a data frame.
        '''
        self.xcols = df.columns - [duration_col, event_col]
        r_df = _convert_to_dataframe(df)
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df

        # Build options string
        options = ''
        if self.minsplit is not None:
            options = ', '.join([options,
                                 'minsplit={}'.format(self.minsplit)])
        if self.minbucket is not None:
            options = ', '.join([options,
                                 'minbucket={}'.format(self.minbucket)])
        if self.xval is not None:
            options = ', '.join([options,
                                 'xval={}'.format(self.xval)])
        if self.cp is not None:
            options = ', '.join([options,
                                 'cp={}'.format(self.cp)])

        if len(options) > 0:
            options = ', control = rpart.control({})'.format(options)

        # Make the command
        cmd = ("myfit = rpart(Surv(r_df${time}, r_df${event}) ~ {incols}, " +
               "data=r_df {options})").format(time=duration_col,
                                              event=event_col,
                                              options=options,
                                              incols='+'.join(self.xcols))
        # Run the command
        self.myfit = r(cmd)
        # Prune it
        if self.cp is not None and self.cp > 0:
            cmd = "myfit <- prune(myfit, cp={})".format(self.cp)
            self.myfit = r(cmd)

        # Now divide into groups for future
        preds = self.predict(df)
        hazards = np.unique(preds)
        # Just to be safe
        hazards.sort()

        # Convert to actual sizes
        highlim = int(self.highlim * df.shape[0])
        lowlim = int(self.lowlim * df.shape[0])

        # Save subgroups here, initialize to outer groups
        self._high = [hazards[-1]]
        self._low = [hazards[0]]

        # Keep track of entire group here for logrank
        high = (preds == hazards[-1])
        low = (preds == hazards[0])

        # Low risk iterates forwards
        for g in hazards[1:]:
            if (np.sum(low) < lowlim or
                not logrank_test(df.loc[low, duration_col],
                                 df.loc[preds==g, duration_col],
                                 df.loc[low, event_col],
                                 df.loc[preds==g, event_col]).is_significant):
                # Append to group
                self._low.append(g)
                low |= (preds == g)
            else:
                break

        # Important to go backwards here of course
        for g in reversed(hazards[:-1]):
            if g in self._low:
                break
            if (np.sum(high) < highlim or
                not logrank_test(df.loc[high, duration_col],
                                 df.loc[preds==g, duration_col],
                                 df.loc[high, event_col],
                                 df.loc[preds==g, event_col]).is_significant):
                # Append to group
                self._high.append(g)
                high |= (preds == g)
            else:
                break
        # Mid is the rest

        # Remember sizes for the benefit of others
        self.high_size = np.sum(high)
        self.low_size = np.sum(low)

    def predict(self, df):
        '''
        Returns an array. Each value will correspond to the group assigned,
        and its hazard? Note that the argument must be a dataframe, but result
        is a numpy array.
        '''
        if self.xcols is None or self.myfit is None:
            raise ValueError("No model has been trained yet!")

        r_df = _convert_to_dataframe(df)
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df

        # Make the command
        cmd = 'predict(myfit, r_df)'
        # Run the command
        prediction = r(cmd)

        # Return np array of it
        return np.array(prediction)

    def predict_classes(self, df):
        '''
        Predict the classes (high, mid, low) of an entire DateFrame.

        Returns a DataFrame.
        '''
        res = pd.DataFrame(index=df.index, columns=['group'])

        preds = self.predict(df)

        for i, p in enumerate(preds):
            if p in self._high:
                res.iloc[i, 0] = 'high'
            elif p in self._low:
                res.iloc[i, 0] = 'low'
            else:
                res.iloc[i, 0] = 'mid'

        return res

    def summary(self):
        '''
        Print R's summary of the model
        '''
        if self.myfit is None:
            raise ValueError('No model has been trained!')
        return r('summary(myfit)')


if __name__ == '__main__':
    # Just a basic test
    import os

    _mayopath = "~/DataSets/vanBelle2009/Therneau2000/data_with_event_only_randomized.csv"
    df = pd.read_csv(os.path.expanduser(_mayopath), sep=None, engine='python')

    cols = ['time', 'event'] + list("trt,age,sex(m=1),ascites,hepato,spiders,edema,bili,chol".split(","))

    df = df.reindex(cols=cols)
    # Remove parenthesis
    c = df.columns.values
    for i, name in enumerate(c):
        if '(' in name:
            c[i] = name[:name.find('(')]
    df.columns = c

    rpart = RPartModel()

    rpart.fit(df, 'time', 'event', minbucket=25)

    preds = rpart.predict(df)
    print(preds)
    print(preds.shape)
    print(np.unique(preds))

    print(rpart.summary())
    print(np.unique(preds))
