# RSTH

<div align="center">



</div>

Official implementation of **"RSTH"**.

This is the code for our RSTH work. After the paper is published, we will upload the core file.

---



## 🌟 Overview


---


### Parameter Descriptions

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `--traffic_file` | Path to traffic matrix file | `./Data/abilene_tm.csv` |
| `--rm_file` | Path to routing matrix file | `./Data/abilene_rm.csv` |
| `--adj_dtw` | DTW-based graph file | `./Data/base2hop_dtw.json` | 
| `--adj_pattern` | Pattern-based graph file | `./Data/base2hop_pattern.json` | 
| `--adj_topo` | Topology-based graph file | `./Data/base2hop_topology.json` | 
| `--seq_len` | Sequence length for input | `144` 
| `--top_k` | Number of neighbors in graph | `20` |
| `--batch_size` | Training batch size | `64` | 
| `--epochs` | Number of training epochs | `90` | 
| `--lr` | Learning rate | `0.001` | Adam optimizer |
| `--seed` | Random seed | `1234321` | Reproducibility |
| `--output_dir` | Output directory | `./Output_RSTH` | 





## Citation

If you find this work useful, please cite our paper:


---

## ⚙️ Requirements


**Main Dependencies:**

- Python 3.8+
- PyTorch 2.0.1+
- NumPy 1.24.3+
- Pandas 1.5.0+
- scikit-learn
- scipy

---

## 📊 Dataset Preparation

### Supported Datasets

| Dataset | Nodes | Flows | Interval | Duration | 
|---------|-------|-------|----------|----------|
| **Abilene** | 12 | 144 | 5 min | 16 weeks | 
