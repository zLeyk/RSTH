import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from Models.st_model import SpatioTemporalEstimator
from Utils.data_utils import get_dataloader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()

parser.add_argument('--traffic_file', type=str, default=r'./Data/abilene_tm.csv')
parser.add_argument('--rm_file', type=str, default=r'./Data/abilene_rm.csv')
# parser.add_argument('--rm_file', type=str, default=r'./Data/abilene_rm_rdm_1.csv')
parser.add_argument('--adj_dtw', type=str, default=r'./Data/base2hop_dtw.json')
parser.add_argument('--adj_pattern', type=str, default=r'./Data/base2hop_pattern.json')
parser.add_argument('--adj_topo', type=str, default=r'./Data/base2hop_topology.json')


parser.add_argument('--model_path', type=str, default='./Output_RSTH/model.pth', help='Path to the trained model (.pth file)')
# parser.add_argument('--model_path', type=str, default='./Output_RSTH/dym_model.pth', help='Path to the trained model (.pth file)')
parser.add_argument('--output_dir', type=str, default='./Output_RSTH',
                    help='Directory to save results')
parser.add_argument('--exp_name', type=str, default='eval_result',
                    help='Experiment name for output files')


parser.add_argument('--seq_len', type=int, default=144)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()



def calculate_metrics(preds, trues):


    min_len = min(len(preds), len(trues))
    preds = preds[:min_len]
    trues = trues[:min_len]

    p_flat = preds.flatten()
    t_flat = trues.flatten()

    epsilon = 1e-9

    # NMAE
    nmae = np.sum(np.abs(p_flat - t_flat)) / (np.sum(np.abs(t_flat)) + epsilon)

    # NRMSE
    nrmse = np.sqrt(np.sum((p_flat - t_flat) ** 2)) / (np.sqrt(np.sum(t_flat ** 2)) + epsilon)

    # R2 Score
    r2 = r2_score(t_flat, p_flat)

    # PCC
    try:
        pcc, _ = pearsonr(t_flat, p_flat)
    except:
        pcc = 0.0
    if np.isnan(pcc):
        pcc = 0.0

    # ARE
    if preds.ndim == 2:
        norms_err_t = np.linalg.norm(preds - trues, axis=1)
        norms_true_t = np.linalg.norm(trues, axis=1)
        norms_true_t[norms_true_t == 0] = epsilon
        are = np.mean(norms_err_t / norms_true_t)
    else:
        are = 0.0

    return {
        'NMAE': nmae,
        'NRMSE': nrmse,
        'R2': r2,
        'PCC': pcc,
        'ARE': are,
    }


def save_results(preds, trues, output_dir, exp_name, metrics):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f'{exp_name}_preds.npy'), preds)
    np.save(os.path.join(output_dir, f'{exp_name}_trues.npy'), trues)
    pd.DataFrame(preds).to_csv(os.path.join(output_dir, f'{exp_name}_preds.csv'),
                               index=False, header=False)
    pd.DataFrame(trues).to_csv(os.path.join(output_dir, f'{exp_name}_trues.csv'),
                               index=False, header=False)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f'{exp_name}_metrics.csv'), index=False)

    print(f"\n[Info] Results and metrics saved to {output_dir}")



def main():
    print("=" * 60)
    print("         Model Evaluation Script")
    print("=" * 60)


    if not os.path.exists(args.model_path):
        print(f"[Error] Model file not found: {args.model_path}")
        return

    print(f"\n[Info] Loading model from: {args.model_path}")
    print(f"[Info] Using device: {args.device}")


    print("\n[Info] Loading test data...")
    train_loader, val_loader, test_loader, scaler, num_links, num_flows, adj_indices = get_dataloader(args)
    adj_indices = adj_indices.to(args.device)

    print(f"[Info] Test samples: {len(test_loader) * args.batch_size}")
    print(f"[Info] Input Links: {num_links}, Output Flows: {num_flows}")

    print("\n[Info] Initializing model...")
    model = SpatioTemporalEstimator(num_links, num_flows, args).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    print(f"[Info] Model loaded successfully")

    print("\n" + "=" * 60)
    print("         Starting Evaluation...")
    print("=" * 60)

    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for y_seq, x in test_loader:
            y_seq, x = y_seq.to(args.device), x.to(args.device)

            pred_x = model(y_seq, adj_indices)

            preds_list.append(pred_x.cpu().numpy())
            trues_list.append(x.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    print(f"[Info] Prediction shape: {preds.shape}")
    print(f"[Info] Ground truth shape: {trues.shape}")

    print("\n[Info] Inverse transforming predictions...")
    preds_inv = scaler.inverse_transform(preds)
    trues_inv = scaler.inverse_transform(trues)

    preds_inv = np.maximum(preds_inv, 0)

    print("\n[Info] Calculating evaluation metrics...")
    metrics = calculate_metrics(preds_inv, trues_inv)

    print("\n" + "=" * 60)
    print("              EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<20} | {'Value':<15}")
    print("-" * 40)
    print(f"{'NMAE':<20} | {metrics['NMAE']:.6f}")
    print(f"{'NRMSE':<20} | {metrics['NRMSE']:.6f}")
    print(f"{'R² Score':<20} | {metrics['R2']:.6f}")
    print(f"{'PCC':<20} | {metrics['PCC']:.6f}")
    print(f"{'ARE':<20} | {metrics['ARE']:.6f}")
    print("=" * 60)

    # save_results(preds_inv, trues_inv, args.output_dir, args.exp_name, metrics)




if __name__ == "__main__":
    main()
