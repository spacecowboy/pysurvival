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
from .cox import _convert_to_dataframe


class RPartModel(object):
    def __init__(self, highlim=0.25, lowlim=0.25, minbucket=25):
        self.highlim = highlim
        self.lowlim = lowlim
        self.minbucket = minbucket
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

        # Make the command
        cmd = ("myfit = rpart(Surv(r_df${time}, r_df${event}) ~ {incols}, " +
               "data=r_df, minbucket={bucket})").format(time=duration_col,
                                                        event=event_col,
                                                        bucket=self.minbucket,
                                                        incols='+'.join(self.xcols))
        print(cmd)
        # Run the command
        self.myfit = r(cmd)

        # Now divide into groups for future
        preds = self.predict(df)
        hazards = np.unique(preds)
        # Just to be safe
        hazards.sort()

        # Convert to actual sizes
        highlim = int(self.highlim * df.shape[0])
        lowlim = int(self.lowlim * df.shape[0])

        self._high = []
        self._low = []

        # Start with high risk, then low risk
        cumsum = 0
        for g in hazards:
            # Append to group
            self._low.append(g)
            cumsum += np.sum(preds == g)
            # Go until the group is a quartile big
            if np.sum(cumsum) >= lowlim:
                break

        # Important to go backwards here of course
        cumsum = 0
        for g in reversed(hazards):
            if g in self._low:
                break
            # Append to group
            self._high.append(g)
            cumsum += np.sum(preds == g)
            # Go until the group is a quartile big
            if np.sum(cumsum) >= highlim:
                break
        # Mid is the rest

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
