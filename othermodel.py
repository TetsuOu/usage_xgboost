import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import numpy as np
from time import time
from dmatrix2np import dmatrix_to_numpy
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()

def create_model(args):
    model = args.model
    if model == 'svr':
        model = SVR()
    elif model == 'knnr':
        model = KNeighborsRegressor(n_neighbors=1, weights='uniform')
    elif model == 'dtr':
        model = DecisionTreeRegressor(max_depth=5)
    elif model == 'sgdr':
        model = SGDRegressor()
    elif model == 'ridge':
        model = Ridge(alpha=0.01)
    elif model == 'rfr':
        model = RandomForestRegressor(n_estimators=20)
    elif model == 'abr':
        model = AdaBoostRegressor(n_estimators=20)
    elif model == 'gbr':
        model = GradientBoostingRegressor(n_estimators=20)
    elif model == 'br':
        model = BaggingRegressor(n_estimators=20)
    else:
        model = LinearRegression()
    return model

model = create_model(args)
print(model)
start_time = time()
data_path = '/data/wangzhe/usage/reagent/xgboost_data'
seed = 42
train_data = os.path.join(data_path,f'train_r{seed}',f'train_r{seed}.buffer')
val_data = os.path.join(data_path,f'val_r{seed}',f'val_r{seed}.buffer')

dtrain = xgb.DMatrix(train_data)
dval = xgb.DMatrix(val_data)

train_feature, train_target = dmatrix_to_numpy(dtrain),dtrain.get_label()
val_feature, val_target = dmatrix_to_numpy(dval),dval.get_label()

print('开始训练：', flush=True)

model.fit(train_feature, train_target)
prediction=model.predict(val_feature)

rmse_std = np.sqrt(mean_squared_error(val_target, prediction))
r2_std = r2_score(val_target, prediction)


print(f' - RSME_std_val: {rmse_std:.4f}, R2_std_val: {r2_std: .4f}')
print(f' 训练测试总耗时: {(time() - start_time) // 60} min', flush=True)