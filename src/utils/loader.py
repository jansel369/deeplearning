import torch as pt
import pandas as pd
# import numpy as np
# import os

def loader(dataset_name, device):
    print("Reading file \"" + dataset_name + "\"....")
    # print(os.getcwd())
    df = pd.read_csv('datasets/' + dataset_name)
    headers = df.columns.tolist()
    values = pt.tensor(df.values, device=device)
    values = values[pt.randperm(values.shape[0])]

    n = values.shape[1] - 1
    m = values.shape[0]
    X = values[:, 0:n]
    Y = values[:, n:n + 1].char().t()

    return {
        "headers": headers,
        "n": n,
        "m": m,
        "X": X,
        "Y": Y,
    }
