import pickle
import datetime
import deap
import pandas as pd

from sklearn.pipeline import Pipeline
from joblib import load
pycaret_model = load("./ml_model/best_model.pkl")


def get_test_data():
    df = pd.read_csv('./test.csv')
    # split source_destination into source and destination
    df = df.assign(source=df.source_destination.str.split(';').str[0])
    df = df.assign(destination=df.source_destination.str.split(';').str[1])
    # split source lat long into source_lat and source_long
    df = df.assign(source_lat=df.source.str.split(',').str[0])
    df = df.assign(source_long=df.source.str.split(',').str[1])
    df = df.assign(destination_lat=df.destination.str.split(',').str[0])
    df = df.assign(destination_long=df.destination.str.split(',').str[1])

    df = df.drop(columns=['source_destination', 'source', 'destination'])
    return df[:10]


def query_model(df):
    res = pycaret_model.predict(df)
    return res

test_data = get_test_data()
res = query_model(test_data)
# print(res)