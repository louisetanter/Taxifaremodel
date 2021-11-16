# imports

from math import remainder
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse, haversine_distance


class Trainer():
    # def __init__(self, X, y):
    def __init__(self, X_train, X_val, y_train, y_val):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        # self.X = X
        # self.y = y
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])

        self.pipeline = pipe
        return self
        # return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)
        # return pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # get data
    N = 100
    df = get_data(nrows=N)

    # clean data
    df = clean_data(df)

    # set X and y

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    trainer = Trainer(X_train, X_val, y_train, y_val)

    trainer.set_pipeline().run()

    # evaluate
    rmse = trainer.evaluate()
    print(rmse)
