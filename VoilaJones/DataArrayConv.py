import pandas as pd
import numpy as np

df=pd.read_csv('Classifier.csv', sep='\t',header=None)

np.save('classifier.npy', df.values)