import xgboost as xgb
import os
import json
from utils import get_RMSE_r2
from time import time

# data_path = '/data/wangzhe/usage/reagent/xgboost_data'
data_path = '/data/wangzhe/usage/reactant_50k/xgboost_data'
seed = 42

train_data = os.path.join(data_path,f'train_r{seed}',f'train_r{seed}.buffer')
val_data = os.path.join(data_path,f'val_r{seed}',f'val_r{seed}.buffer')

dtrain = xgb.DMatrix(train_data)
dval = xgb.DMatrix(val_data)

parameters = {
                'max_depth': 15,#15
                'learning_rate': 0.02,
                'min_child_weight': 10,
                'max_delta_step': 1,
                'subsample': 0.85, #0.85
                'colsample_bytree': 0.9,
                'reg_alpha': 0,
                'reg_lambda': 0.2,
                'scale_pos_weight': 0.4,
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            }

with open('configs/bst_params.json', "w", encoding="utf-8") as f:
    f.write(json.dumps(parameters, ensure_ascii=False, indent=2))

start_time = time()

num_rounds = 2000
print("训练模型", flush=True)
bst = xgb.train(parameters, dtrain, num_rounds)
print("保存模型中", flush=True)
bst.save_model('models/bst.model')
bst.dump_model('models/dump.raw.txt')


rmse_std_train, r2_std_train = get_RMSE_r2(model=bst, data=dtrain)
rmse_std_val, r2_std_val = get_RMSE_r2(model=bst, data=dval)

print(f' - RSME_std_train: {rmse_std_train:.4f}, R2_std_train: {r2_std_train: .4f}')
print(f' - RSME_std_test: {rmse_std_val:.4f}, R2_std_test: {r2_std_val: .4f}')
print(f' 训练测试总耗时: {(time() - start_time) // 60} min', flush=True)
