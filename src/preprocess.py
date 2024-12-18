import pandas as pd

df = pd.read_csv('data/iris.csv')
df['target'] = pd.Categorical(df['species']).codes
df.drop(columns=['species'], inplace=True)
df.to_csv('data/processed_iris.csv', index=False)
