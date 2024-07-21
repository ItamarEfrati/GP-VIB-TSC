import os
import requests
from tqdm import tqdm
import numpy as np
import torch
from lightning_fabric import seed_everything
from sklearn.preprocessing import StandardScaler


def set_seed(config):
    if config.get("seed"):
        seed = config.seed
        seed_everything(config.seed, workers=True)
    else:
        rand_bytes = os.urandom(4)
        seed = int.from_bytes(rand_bytes, byteorder='little', signed=False)
        seed_everything(seed, workers=True)
    return seed


def download_file(url, local_file_path):
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    response = requests.get(url, stream=True, headers=headers)
    content_length = int(response.headers['Content-Length'])
    pbar = tqdm(total=content_length)
    with open(local_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=max(20_000_000, content_length)):
            if chunk:
                f.write(chunk)
            pbar.update(len(chunk))


def download_files(data_urls, download_dir):
    local_files = []
    for url in data_urls:
        file_name = url.split('/')[-1].split('?')[0]
        local_file_path = os.path.join(download_dir, file_name)
        local_files.append(local_file_path)
        if os.path.exists(local_file_path):
            continue
        download_file(url, local_file_path)
    return local_files


def scale_data(data_df, train_users, val_users, test_users):
    ss = StandardScaler()
    train_df = data_df.loc[train_users].copy()
    X_train = ss.fit_transform(train_df.values)
    X_train = np.nan_to_num(X_train, nan=0)
    train_df[:] = X_train

    val_df = data_df.loc[val_users].copy()
    X_val = ss.transform(val_df.values)
    X_val = np.nan_to_num(X_val, nan=0)
    val_df[:] = X_val

    test_df = data_df.loc[test_users].copy()
    X_test = ss.transform(test_df.values)
    X_test = np.nan_to_num(X_test, nan=0)
    test_df[:] = X_test

    return train_df, val_df, test_df


def pandas_to_tensors(df, users):
    data = df.groupby('record_id') \
        .apply(lambda g: torch.tensor(g.values, dtype=torch.float)) \
        .loc[users].values

    return torch.stack(list(data))
