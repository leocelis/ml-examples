import numpy as np
import pandas as pd

c = "winemag-data-130k-v2.csv"
df = pd.read_csv(c, low_memory=False)
#df = pd.read_csv(c, low_memory=False, nrows=100)
print(f"{len(df.index)} total entries")

df_it = df.query("country == 'Italy'")
df_ar = df.query("country == 'Argentina'")

it_avg_points = df_it['points'].mean()
ar_avg_points = df_ar['points'].mean()

print(f"Italy: {it_avg_points} - Argentina: {ar_avg_points}")
