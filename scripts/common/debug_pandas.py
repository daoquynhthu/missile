import pandas as pd
df = pd.read_csv('temp_traj.csv')
print(df.dtypes)
print(df.head())
print(df['X(m)'].iloc[0], type(df['X(m)'].iloc[0]))
