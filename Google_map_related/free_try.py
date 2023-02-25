import pandas as pd
import requests
import json
import time

df = pd.read_csv('route_working_mapping.csv')
df['source_lan'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[0])
df['source_lon'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[1])
df['dest_lan'] = df['source_destination'].map(lambda x:x.split(';')[1].split(',')[0])
df['dest_lon'] = df['source_destination'].map(lambda x:x.split(';')[1].split(',')[1])
# df = df.drop(['source_destination'], axis=1)
df['google_duration'] = ''
print(df.head())

count = 0
key = "AIzaSyBN9wjKeVnXXeMK3dtVWFgFYjGfL18MyyA"
direction_url = "https://maps.googleapis.com/maps/api/directions/json?origin={},{}&destination={},{}&mode=driving&key={}"
payload={}
headers = {}
for index, row in df.iterrows():
    print(direction_url.format(row['source_lan'], row['source_lon'], row['dest_lan'], row['dest_lon'], key))
    response = requests.get(direction_url.format(row['source_lan'], row['source_lon'], row['dest_lan'], row['dest_lon'], key))
    result_json_obj = json.loads(response.text)
    print(result_json_obj['routes'][0]['legs'][0]['duration']['value'])
    df.loc[index,'google_duration'] = result_json_obj['routes'][0]['legs'][0]['duration']['value']
    time.slee(500)
    count+=1
    if count>10:
        break


df.to_csv("route_working_mapping_with_google.csv")