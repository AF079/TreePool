from re import T
import treelite.python.treelite as treelite 
#import treelite.runtime.python.treelite_runtime
import treelite_runtime
import time

import tl2cgen.python.tl2cgen as tl2cgen
from sklearn.datasets import load_iris
import xgboost as xgb



X,y = load_iris(return_X_y=True)

print(f"dimensions of X = {X.shape}")

print(f"dimensions of y = {y.shape}")

dtrain = xgb.DMatrix(X,label=y)

params = {"max_depth":3,"eta":0.1,"objective":"multi:softprob",
          "eval_metric":"mlogloss","num_class":3}

bst = xgb.train(params,dtrain,num_boost_round=20,evals=[(dtrain,'train')])
model = treelite.Model.from_xgboost(bst)
Z = X[0:149,:]
toolchain = 'gcc'

def test_tl2cgen(n):
    tl2cgen.export_lib(model,toolchain=toolchain,libpath='genc/mymodel.so',params={})
    #tl2cgen.generate_c_code(model, dirpath="genc/model",params={})
    predictor = tl2cgen.Predictor("genc/mymodel.so")
    dmat = tl2cgen.DMatrix(Z)
    for i in range(n):
        out_pred_tl2cgen = predictor.predict(dmat,verbose=True)


def test_tl(n):
    model.export_lib(toolchain=toolchain, libpath='./mymodel.so', verbose=True)
    predictor_tl = treelite_runtime.Predictor('./mymodel.so', verbose=True)
    dmat = treelite_runtime.DMatrix(Z)
    total_time = 0
    for i in range(n):
        out_pred_tl = predictor_tl.predict(dmat,verbose=True)


test_tl2cgen(10)
#test_tl(10)


  # total_size = tl2cgen::detail::threading_utils::custom_tpool::Custom_ParallelFor_ThreadPool(
  #     std::size_t(0), nthread, dmat, row_ptr, out_result, pred_margin,pred_func_.get());