import torch as pt
import pandas as pd

def loader(dataset_name, device):
    print("Reading file \"" + dataset_name + "\"....")
    df = pd.read_csv('../datasets/' + dataset_name)
    headers = df.columns.tolist()
    values = df.values
    n = values.shape[1] - 1
    m = values.shape[0]
    X = pt.tensor(values[:, 0:n], device=device)
    Y = pt.tensor(values[:, n:n + 1].T, device=device)

    return {
        "headers": headers,
        "n": n,
        "m": m,
        "X": X,
        "Y": Y,
    }