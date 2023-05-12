import pandas as pd
import os
from pathlib import Path
from utils import create_feature
import xgboost as xgb
import shutil

# path = '/data/wangzhe/usage/reagent/raw_data/output'
path = '/data/wangzhe/usage/reactant_50k/raw_data'

seed = 42

dirs = [f'val_r{seed}', f'test_r{seed}', f'train_r{seed}']

# save_path = '/data/wangzhe/usage/reagent/xgboost_data'
save_path = '/data/wangzhe/usage/reactant_50k/xgboost_data'

for subdir in dirs:
    csvfile = os.path.join(path, f'{subdir}.csv')
    df = pd.read_csv(csvfile)
    print(f'{subdir}:')
    print(f'dataset size: {df.shape[0]}')
    feat, label = create_feature(df)

    dmatrix = xgb.DMatrix(feat, label=label)

    to_path = os.path.join(save_path, subdir)
    for mpath in [to_path]:
        if os.path.exists(mpath) and 'y'=="y":
            shutil.rmtree(mpath)
        os.makedirs(mpath)
    
    dmatrix.save_binary(os.path.join(to_path,f'{subdir}.buffer'))

print('success')