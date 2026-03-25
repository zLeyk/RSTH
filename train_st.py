import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import time
import os
import random
import pandas as pd

from Utils.data_utils import get_dataloader
from Models.st_model import SpatioTemporalEstimator
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ================= 配置参数 =================
parser = argparse.ArgumentParser()

parser.add_argument('--traffic_file', type=str, default=r'./Data/abilene_tm.csv')
parser.add_argument('--rm_file', type=str, default=r'./Data/abilene_rm.csv')
# parser.add_argument('--rm_file', type=str, default=r'./Data/abilene_rm_dym.csv')


parser.add_argument('--adj_dtw', type=str, default=r'./Data/base2hop_dtw.json')
parser.add_argument('--adj_pattern', type=str, default=r'./Data/base2hop_pattern.json')
parser.add_argument('--adj_topo', type=str, default=r'./Data/base2hop_topology.json')
parser.add_argument('--top_k', type=int, default=20)

parser.add_argument('--output_dir', type=str, default='./Output_RSTH', help='Directory to save model and results')
parser.add_argument('--exp_name', type=str, default='exp_rm', help='Experiment name for file prefix')

parser.add_argument('--seq_len', type=int, default=144)  #

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=1234321, help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--head_type', type=str, default='linear')
parser.add_argument('--tensor_rank', type=int, default=8)
parser.add_argument('--ablation', type=str, default='none',
                    choices=['none', 'no_temp', 'no_refine',  'no_gate'],
                    help='Ablation study mode')
args = parser.parse_args()


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或 ':16:8'
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass



def save_results(preds, trues, output_dir, exp_name):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f'{exp_name}_preds.npy'), preds)
    np.save(os.path.join(output_dir, f'{exp_name}_trues.npy'), trues)

    pd.DataFrame(preds).to_csv(os.path.join(output_dir, f'{exp_name}_preds.csv'), index=False, header=False)
    pd.DataFrame(trues).to_csv(os.path.join(output_dir, f'{exp_name}_trues.csv'), index=False, header=False)

    print(f"[Info] Results saved to {output_dir}")


def main():

    # seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Spatio-Temporal TM Estimation (R-STH) ===")
    print(f"Output Directory: {args.output_dir}")

    train_loader, val_loader, test_loader, scaler, num_links, num_flows, adj_indices = get_dataloader(args)

    adj_indices = adj_indices.to(args.device)
    print(f"Input Links: {num_links}, Output Flows: {num_flows}")

    seed_everything(args.seed)
    model = SpatioTemporalEstimator(num_links, num_flows, args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, f'{args.exp_name}_best_model.pth')

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        t0 = time.time()

        for y_seq, x in train_loader:
            y_seq, x = y_seq.to(args.device), x.to(args.device)

            optimizer.zero_grad()
            pred_x = model(y_seq, adj_indices)
            loss = criterion(pred_x, x)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())


        model.eval()
        val_loss = []
        with torch.no_grad():
            for y_seq, x in val_loader:
                y_seq, x = y_seq.to(args.device), x.to(args.device)
                pred_x = model(y_seq, adj_indices)
                loss = criterion(pred_x, x)
                val_loss.append(loss.item())

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (Val Loss: {avg_val_loss:.9f})")
            print(f"epoch: {epoch}")
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {time.time() - t0:.1f}s")

    print("\nStarting Testing with Best Model...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    preds_list, trues_list = [], []

    with torch.no_grad():
        for y_seq, x in test_loader:
            y_seq, x = y_seq.to(args.device), x.to(args.device)
            pred_x = model(y_seq, adj_indices)
            preds_list.append(pred_x.cpu().numpy())
            trues_list.append(x.cpu().numpy())


    preds = np.concatenate(preds_list, axis=0)  # [Test_Samples, Nodes]
    trues = np.concatenate(trues_list, axis=0)  # [Test_Samples, Nodes]

    preds_inv = scaler.inverse_transform(preds)
    trues_inv = scaler.inverse_transform(trues)


    preds_inv = np.maximum(preds_inv, 0)

    save_results(preds_inv, trues_inv, args.output_dir, args.exp_name)


    nmae = np.sum(np.abs(preds_inv - trues_inv)) / np.sum(np.abs(trues_inv))
    nrmse = np.sqrt(np.sum((preds_inv - trues_inv) ** 2)) / np.sqrt(np.sum(trues_inv ** 2))


    preds_flat = preds_inv.flatten()
    trues_flat = trues_inv.flatten()
    ss_res = np.sum((trues_flat - preds_flat) ** 2)
    ss_tot = np.sum((trues_flat - np.mean(trues_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n=== Final R-STH Results ===")
    print(f"NMAE:  {nmae:.4f}")
    print(f"NRMSE: {nrmse:.4f}")
    print(f"R2:    {r2:.4f}")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()