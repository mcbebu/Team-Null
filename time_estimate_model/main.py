# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


# input hourly time format: "2022-11-02T00:00"
# output rain format (mm): 0.2


# input: source_longitude, source_latitude, destination_longitude, destination_latitude
# output: estimated time to travel from source to destination
def knn_regressor_a_b_lat_long_estimated_time_lookup_table():
    df = pd.read_csv('route_working_mapping.csv')
    X_str = df[['source_destination']].values
    # print(X_str[:10])

    # split source_destination into four columns
    # source_longitude, source_latitude, destination_longitude, destination_latitude
    X = [ x_one_str[0].replace(';', ',').split(',') for x_one_str in X_str]
    X = [[float(num) for num in row] for row in X]
    print(X[:10])

    y = df['time_diff'].values
    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(X, y)

    print('gt', y[0])
    print('prediction', neigh.predict([[1.386478068, 103.7602565, 1.387284982, 103.7590052]]))


def weather_lookup_table(hourly_time):
    df = pd.read_csv('time_rain.csv').values
    hour_rain_dic = dict(df)
    rain_volumn = hour_rain_dic[hourly_time]


def main():

    df = pd.read_csv('route_working_mapping.csv')
    X_str = df[['source_destination']].values
    # print(X_str[:10])

    # split source_destination into four columns:
    # source_longitude, source_latitude, destination_longitude, destination_latitude
    X = [ x_one_str[0].replace(';', ',').split(',') for x_one_str in X_str]
    # print(X[:10])

    y = df['time_diff'].values
    # print('y', y[:10])
    data_len = len(X)
    train_len = int(len(X) * 0.8)
    clf = RandomForestRegressor(max_depth=2, random_state=0)
    clf.fit(X[:train_len], y[:train_len])

    print('gt', y[0])
    # print('prediction', clf.predict(['1.386478068,103.7602565;1.387284982,103.7590052'.replace(';', ',').split(',')]))
    print('prediction', clf.predict([X[0]]))
    print('mean squared error', mean_squared_error(y[train_len:], clf.predict(X[train_len:])))
    print('r2 score', r2_score(y[train_len:], clf.predict(X[train_len:])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    knn_regressor_a_b_lat_long_estimated_time_lookup_table()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
