# -*- coding: utf-8 -*-
"""
Wraps the RPart model.
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
#from rpy2.robjects.vectors import FloatVector
# R imports
__rpart = importr('rpart')
__survival = importr('survival')

import numpy as np
import pandas as pd
import pandas.rpy.common as com


class RPartModel(object):
    def __init__(self):
        self.xcols = None
        self.myfit = None

    def fit(self, df, duration_col, event_col):
        self.xcols = df.columns - [duration_col, event_col]
        r_df = com.convert_to_r_dataframe(df)
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df

        # Make the command
        cmd = ("myfit = rpart(Surv(r_df${time}, r_df${event}) ~ {incols}, " +
               "data=r_df, minbucket=10)").format(time=duration_col,
                                                  event=event_col,
                                                  incols='+'.join(self.xcols))

        # Run the command
        self.myfit = r(cmd)

    def predict(self, df):
        if self.xcols is None or self.myfit is None:
            raise ValueError("No model has been trained yet!")

        r_df = com.convert_to_r_dataframe(df[self.xcols])
        # Insert into namespace
        robjects.globalenv['r_df'] = r_df
        # Make the command
        cmd = 'predict(myfit, r_df)'
        # Run the command
        prediction = r(cmd)

        # Return np array of it
        return np.array(prediction)

    def summary(self):
        '''
        Print R's summary of the model
        '''
        if self.model is None:
            raise ValueError('No model has been trained!')
        print(r('summary(myfit)'))




if __name__ == '__main__':
    import os

    _mayopath = "~/DataSets/vanBelle2009/Therneau2000/data_with_event_only_randomized.csv"
    _mayo = dict(filename=os.path.expanduser(_mayopath),
                 timecol='time',
                 eventcol='event',
                 xcols="trt,age,sex(m=1),ascites,hepato,spiders,edema,bili,chol".split(","))
    df = pd.read_csv(_mayopath, sep=None, engine='python')

    cols = ['time', 'event'] + list("trt,age,sex(m=1),ascites,hepato,spiders,edema,bili,chol".split(","))

    df = df.reindex(cols=cols)
