import pandas as pd

df = pd.read_csv('route_working_mapping.csv')
df['source_lan'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[0])
df['source_lon'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[1])
df['dest_lan'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[0])
df['dest_lon'] = df['source_destination'].map(lambda x:x.split(';')[0].split(',')[1])
df = df.drop(['source_destination'], axis=1)
print(df.head())

count = 0
for row in df.iterrows():
    print(row)
    count+=1
    if count>2:
        break