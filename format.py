import os
import pandas as pd

LENGTH = 10
BREADTH = 5
SIZE = LENGTH*BREADTH

src = './data/img/'
dst = f'./data/{SIZE}_data.csv'
dst_catcol = f'{dst}.catcol'

whole = []

for file in sorted(os.listdir(src)):
    df = pd.read_csv(os.path.join(src, file))
    lis = df['0'].tolist()
    if len(lis) >= SIZE:
        # whole.append(lis[-SIZE:].reverse())
        whole.append(lis[-SIZE-165:-SIZE])
        

df = pd.DataFrame(whole)
df.to_csv(dst, index = False)

classi = [i for i in range(SIZE)]
df = pd.DataFrame(classi)
df.to_csv(dst_catcol, index = False, header = False)