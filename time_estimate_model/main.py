# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
import pandas as pd

def main():

    df = pd.read_csv('route_working_mapping.csv')
    X_str = df[['source_destination']].values
    print(X_str[:10])

    # TODO split source_destination into four columns, source_longitude, source_latitude,
    X = [ x_one_str[0].replace(';', ',').split(',') for x_one_str in X_str]
    print(X[:10])

    y = df['time_diff'].values
    print('y', y[:10])
    clf = RandomForestRegressor(max_depth=2, random_state=0)
    clf.fit(X, y)

    print('gt', y[0])
    # print('prediction', clf.predict(['1.386478068,103.7602565;1.387284982,103.7590052'.replace(';', ',').split(',')]))
    print('prediction', clf.predict([X[0]]))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
