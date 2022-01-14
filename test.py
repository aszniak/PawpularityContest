import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')

popularity = df['Pawpularity']
popularity = popularity.to_numpy()


targets = np.ones(len(popularity)) * popularity.mean()
rmse = np.sqrt(((popularity - targets)**2).mean())
print(rmse)
