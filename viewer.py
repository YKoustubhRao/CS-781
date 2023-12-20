import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


src = './data/50_data.csv'
csv = pd.read_csv(src)
need = [3]
N = len(csv.index)



for i in need:
    data = np.array(csv.iloc[i].tolist()).reshape((10,5))
    plt.imshow(data, interpolation='nearest')
    plt.savefig(f'./data/image/{i}.jpg')
    plt.show()
    break

