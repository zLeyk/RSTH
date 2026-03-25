import torch
import numpy as np
import pandas as pd
import json
import os
from torch.utils.data import Dataset, DataLoader


class TimeSeriesTMEDataset(Dataset):
    def __init__(self, tm_filepath, rm_filepath, train_size, test_size, period='train', scale=1e9, seq_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.period = period

        self.traffic, self.link, self.rm = self.read_data(tm_filepath, rm_filepath, train_size, test_size, scale)

        if period == 'test':
            self.traffic = self.traffic[train_size - seq_len + 1:, :]
            self.link = self.link[train_size - seq_len + 1:, :]
        else:
            self.traffic = self.traffic[:train_size, :]
            self.link = self.link[:train_size, :]

        self.len = self.traffic.shape[0] - seq_len + 1
        self.dim_flow = self.traffic.shape[1]
        self.dim_link = self.link.shape[1]

    def read_data(self, tm_filepath, rm_filepath, train_size, test_size, scale):

        df = pd.read_csv(tm_filepath, header=None)


        if 'abilene' in tm_filepath.lower():
            df.drop(df.columns[-1], axis=1, inplace=True)
        else:
            print(f"[Warning] Unknown dataset: {tm_filepath}")

        print(f"Raw traffic data - Min: {df.values.min()}, Max: {df.values.max()}, Mean: {df.values.mean()}")

        total_len = train_size + test_size
        if len(df) < total_len: total_len = len(df)

        traffic = df.values[:total_len] / scale

        print(f"Scaled traffic data - Min: {traffic.min()}, Max: {traffic.max()}, Mean: {traffic.mean()}")


        rm_df = pd.read_csv(rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)
        rm = rm_df.values


        if traffic.shape[1] != rm.shape[0]:
            if traffic.shape[1] == rm.shape[1]:
                rm = rm.T
            else:
                raise ValueError(f"Shape mismatch: TM {traffic.shape}, RM {rm.shape}")

        link = np.matmul(traffic, rm)

        print(f"Link data - Min: {link.min()}, Max: {link.max()}, Mean: {link.mean()}")

        return torch.FloatTensor(traffic), torch.FloatTensor(link), torch.FloatTensor(rm)

    def __getitem__(self, ind):
        y_seq = self.link[ind: ind + self.seq_len]
        x_target = self.traffic[ind + self.seq_len - 1]
        return y_seq, x_target

    def __len__(self):
        return max(0, self.len)


class FlowTMScaler:
    def __init__(self, scale=1e9):
        self.scale = scale

    def inverse_transform(self, data):
        return data * self.scale


def load_graph_data(graph_files, num_nodes, top_k):
    adj_indices = []
    for file_path in graph_files:
        if not os.path.exists(file_path):
            print(f"[Warning] Graph file {file_path} not found. Using self-loop.")
            indices = torch.arange(num_nodes).unsqueeze(1).repeat(1, top_k)
        else:
            print(f"Loading graph: {file_path}")
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
            indices = torch.zeros((num_nodes, top_k), dtype=torch.long)
            for node_idx, info in graph_data.items():
                idx = int(node_idx)
                if idx >= num_nodes: continue
                neighbors = info['hop1']
                if len(neighbors) >= top_k:
                    neighbors = neighbors[:top_k]
                else:
                    neighbors = neighbors + [idx] * (top_k - len(neighbors))
                indices[idx] = torch.tensor(neighbors)
        adj_indices.append(indices)
    return torch.stack(adj_indices, dim=0)


class LogMaxScaler:
    def __init__(self, max_val=None):
        self.max_log = np.log1p(max_val) if max_val else 1.0

    def transform(self, data):
        return np.log1p(data) / self.max_log

    def inverse_transform(self, data):
        return np.expm1(data * self.max_log)

def get_dataloader(args):

    dataset_name = 'abilene'
    train_weeks = 15
    test_weeks = 1
    samples_per_week = 2016
    scale = 1e9
    print(f"[Info] Detected Abilene dataset. Using scale={scale}, train_weeks={train_weeks}")

    train_size = train_weeks * samples_per_week
    test_size = test_weeks * samples_per_week
    print(f"[Info] Dataset size: train={train_size}, test={test_size}")

    train_dataset = TimeSeriesTMEDataset(
        args.traffic_file, args.rm_file,
        train_size, test_size,
        period='train', scale=scale, seq_len=args.seq_len
    )
    test_dataset = TimeSeriesTMEDataset(
        args.traffic_file, args.rm_file,
        train_size, test_size,
        period='test', scale=scale, seq_len=args.seq_len
    )

    print(
        f"Train dataset link data - Min: {train_dataset.link.min().item()}, Max: {train_dataset.link.max().item()}, Mean: {train_dataset.link.mean().item()}")
    print(
        f"Train dataset traffic data - Min: {train_dataset.traffic.min().item()}, Max: {train_dataset.traffic.max().item()}, Mean: {train_dataset.traffic.mean().item()}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    scaler = FlowTMScaler(scale)



    graph_files = [args.adj_dtw, args.adj_pattern, args.adj_topo]
    adj_indices = load_graph_data(graph_files, train_dataset.dim_flow, args.top_k)

    return train_loader, test_loader, test_loader, scaler, train_dataset.dim_link, train_dataset.dim_flow, adj_indices