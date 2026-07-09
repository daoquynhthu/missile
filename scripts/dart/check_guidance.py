import pandas as pd
df = pd.read_csv('dart_mc_results.csv')
active_rate = df['Guidance'].mean() * 100
print(f"Guidance Active Rate: {active_rate:.2f}%")
print(df['Guidance'].value_counts())
