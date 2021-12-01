import numpy as np
import pandas as pd

data = pd.read_csv('GlobalTerrorismReport-2015.csv')

print(data.columns)

print(data.info())

print(data.describe())

df = data["country"]
dff = df.value_counts()
print(dff)