import treelite.python.treelite as treelite 
#import treelite.runtime.python.treelite_runtime
import treelite.runtime.python.treelite_runtime as treelite_runtime
import time

import tl2cgen.python.tl2cgen as tl2cgen
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import xgboost as xgb

import numpy as np
builder = treelite.ModelBuilder(num_feature=3)
tree = treelite.ModelBuilder.Tree()
tree[0].set_numerical_test_node(
        feature_id=0,
        opname="<",
        threshold=5.0,
        default_left=True,
        left_child_key=1,
        right_child_key=2
        )
tree[1].set_leaf_node(0.6)
tree[2].set_leaf_node(-0.4)
tree[0].set_root()
builder.append(tree)
model = builder.commit()  # Obtain treelite.Model object
input = np.random.rand(1,2)

def test_rt(cnt):
    so_name = "tmp.so"
    model.export_lib(toolchain='gcc',
            libpath=so_name, verbose=True, params={'parallel_comp': 1})
    p = treelite_runtime.Predictor(so_name)
    dmat = treelite_runtime.DMatrix(input)
    start = time.time()
    for _ in range(cnt):
        ret = p.predict(dmat)
    end = time.time()
    print(f'time consumed for {cnt} treelite_runtime predictions : {end-start}')
    print(ret)

def test_tl2cgen(cnt):
    so_name = "tmp1.so"
    tl2cgen.export_lib(model, toolchain='gcc',
            libpath=so_name, verbose=True, params={'parallel_comp': 1})
    p = tl2cgen.Predictor(so_name)
    dmat = tl2cgen.DMatrix(input)
    start = time.time()
    for _ in range(cnt):
        ret = p.predict(dmat)
    end = time.time()
    print(f'time consumed for {cnt} tl2cgen predictions : {end-start}')
    print(ret)
#test_rt(1)
test_tl2cgen(1)