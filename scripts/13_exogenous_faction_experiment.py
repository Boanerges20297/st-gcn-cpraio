"""Aggregate faction signals and exogenous seizure/homicide features, run comparative experiments.

Experiments:
 - baseline
 - +procedimentos
 - +procedimentos + faccao (one-hot of major factions)
 - +procedimentos + faccao + exogenos (homicidios, drogas_sum, armas_sum, dinheiro_sum)

Outputs:
 - outputs/exog_faction_summary.csv
 - outputs/exog_faction_per_bairro_rmse.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.stgcn import STGCNModel


def load_json_table(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    for item in raw:
        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
            return pd.DataFrame(item['data'])
    return pd.DataFrame(raw)


def load_mapping_resources():
    report = json.load(open('outputs/neighborhood_mapping_report.json', 'r', encoding='utf-8'))
    mapping_cache = report.get('mapping_cache', {})
    params = json.load(open('data/processed/normalization_params_deduplicated.json', 'r', encoding='utf-8'))
    neighborhood_map = params['neighborhood_mapping']
    date_range = params['date_range']
    return mapping_cache, neighborhood_map, date_range


def map_to_bairro_id(raw_name, mapping_cache, neighborhood_map):
    # try mapping_cache -> standardized -> bairro_id
    std = None
    if raw_name in mapping_cache and mapping_cache[raw_name]:
        std = mapping_cache[raw_name]['matched']
    if not std:
        try:
            std = str(raw_name).upper()
        except Exception:
            return None
    return neighborhood_map.get(std)


def aggregate_exogenous(df_ops, mapping_cache, neighborhood_map, date_range):
    df = df_ops.copy()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df[df['Data'].notna()].copy()
    df['date_norm'] = df['Data'].dt.normalize()

    # map bairro
    df['std_name'] = df['BairroOcor'].apply(lambda x: mapping_cache.get(x, {}).get('matched') if x in mapping_cache else None)
    df['std_name'] = df['std_name'].fillna(df['BairroOcor'].str.upper())
    # map to id
    nm_upper = {k.upper(): v for k, v in neighborhood_map.items()}
    df['bairro_id'] = df['std_name'].map(nm_upper)
    df = df[df['bairro_id'].notna()].copy()
    df['bairro_id'] = df['bairro_id'].astype(int)

    # homicide flag
    df['is_homicidio'] = df['Natureza'].fillna('').str.lower().str.contains('homicid')

    # numeric conversions
    df['total_armas'] = pd.to_numeric(df.get('total_armas_cache', 0), errors='coerce').fillna(0)
    df['total_drogas'] = pd.to_numeric(df.get('total_drogas_cache', 0), errors='coerce').fillna(0)
    df['dinheiro'] = pd.to_numeric(df.get('Dinheiro_Apreendido', 0), errors='coerce').fillna(0)

    # date index
    start = pd.to_datetime(date_range['start']).normalize()
    end = pd.to_datetime(date_range['end']).normalize()
    dates = pd.date_range(start, end, freq='D')
    D = len(dates)
    N = max(neighborhood_map.values()) + 1

    # initialize arrays
    homic = np.zeros((D, N), dtype=float)
    armas = np.zeros((D, N), dtype=float)
    drogas = np.zeros((D, N), dtype=float)
    dinheiro = np.zeros((D, N), dtype=float)

    date_to_idx = {d.normalize(): i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        d = row['date_norm']
        if d not in date_to_idx:
            continue
        i = date_to_idx[d]
        j = int(row['bairro_id'])
        homic[i, j] += int(row['is_homicidio'])
        armas[i, j] += float(row['total_armas'])
        drogas[i, j] += float(row['total_drogas'])
        dinheiro[i, j] += float(row['dinheiro'])

    return dates, {'homic': homic, 'armas': armas, 'drogas': drogas, 'dinheiro': dinheiro}


def aggregate_faction_by_bairro(mapping_cache, neighborhood_map):
    # facoes_territorio.csv has local_oficial and faccao_predominante
    df = pd.read_csv('data/graph/facoes_territorio.csv')
    nm_upper = {k.upper(): v for k, v in neighborhood_map.items()}
    # normalize local_oficial to uppercase and map
    df['local_up'] = df['local_oficial'].str.upper()
    df['bairro_id'] = df['local_up'].map(nm_upper)
    df = df[df['bairro_id'].notna()].copy()
    df['bairro_id'] = df['bairro_id'].astype(int)

    # major factions in dataset
    factions = df['faccao_predominante'].unique().tolist()
    factions = [str(x) for x in factions if pd.notna(x)]

    N = max(neighborhood_map.values()) + 1
    one_hot = np.zeros((N, len(factions)), dtype=float)
    faction_to_idx = {f: i for i, f in enumerate(factions)}
    for _, r in df.iterrows():
        bid = int(r['bairro_id'])
        f = r['faccao_predominante']
        one_hot[bid, faction_to_idx[f]] = 1.0

    return factions, one_hot


def create_windows_and_train(X, A, epochs=25, window_size=7, horizon=1):
    # reuse earlier training approach but return per-node RMSE
    def create_temporal_windows(X, window_size=7, horizon=1):
        windows = []
        targets = []
        T = X.shape[0]
        for t in range(T - window_size - horizon + 1):
            windows.append(X[t:t+window_size])
            targets.append(X[t+window_size+horizon-1])
        if len(windows) == 0:
            return np.zeros((0, window_size) + X.shape[1:]), np.zeros((0,) + X.shape[1:])
        return np.array(windows), np.array(targets)

    Xw, yt = create_temporal_windows(X, window_size, horizon)
    split = int(len(Xw) * 0.8)
    train_ds = WindowDataset(Xw[:split], yt[:split])
    val_ds = WindowDataset(Xw[split:], yt[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    T, N, F = X.shape[0], X.shape[1], X.shape[2]
    model = STGCNModel(in_channels=F, hidden_channels=64, out_channels=F, A=A)
    device = torch.device('cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    best_val = float('inf'); best_preds = None; best_targs = None
    for ep in range(1, epochs + 1):
        model.train(); total=0.0
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); out=model(xb); loss=crit(out,yb); loss.backward(); opt.step()
            total += loss.item()*xb.size(0)
        val_loss = 0.0
        preds=[]; targs=[]
        model.eval()
        with torch.no_grad():
            for xb,yb in val_loader:
                xb=xb.to(device); yb=yb.to(device)
                out=model(xb)
                val_loss += crit(out,yb).item()*xb.size(0)
                preds.append(out.cpu().numpy()); targs.append(yb.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {ep}/{epochs} — val_loss: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_preds = np.vstack(preds)
            best_targs = np.vstack(targs)

    # compute per-node RMSE
    mse_per_node_feat = ((best_preds - best_targs) ** 2).mean(axis=0)
    rmse_per_node = np.sqrt(mse_per_node_feat.mean(axis=1))
    overall_rmse = np.sqrt(((best_preds - best_targs) ** 2).mean())
    return overall_rmse, rmse_per_node


class WindowDataset(Dataset):
    def __init__(self, X_windows, y_targets):
        self.X = X_windows; self.y = y_targets
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()


def main():
    mapping_cache, neighborhood_map, date_range = load_mapping_resources()
    df_ops = load_json_table('data/raw/ocorrencia_policial_operacional.json')
    dates, exog = aggregate_exogenous(df_ops, mapping_cache, neighborhood_map, date_range)

    # save aggregated exogenous arrays
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/exog_homic.npy', exog['homic'])
    np.save('outputs/exog_armas.npy', exog['armas'])
    np.save('outputs/exog_drogas.npy', exog['drogas'])
    np.save('outputs/exog_dinheiro.npy', exog['dinheiro'])

    factions, faction_onehot = aggregate_faction_by_bairro(mapping_cache, neighborhood_map)
    np.save('outputs/faction_onehot.npy', faction_onehot)
    print('Factions found (after mapping):', factions)

    # determine bairro with highest cumulative homicides
    cumulative_homic = exog['homic'].sum(axis=0)
    target_bairro = int(np.nanargmax(cumulative_homic))
    # build reverse mapping id->name if available
    rev_map = {v: k for k, v in neighborhood_map.items()}
    target_name = rev_map.get(target_bairro, None)
    print(f"Selected bairro for detailed study: id={target_bairro}, name={target_name}")

    # load base tensor and adjacency
    X = np.load('data/processed/node_feature_tensor.npy')  # (T,N,F)
    A = np.load('data/processed/adjacency_matrix.npy')

    # load previously computed procedures normalized
    proc = np.load('outputs/procedures_daily_node_matrix.npy')
    p99 = np.percentile(proc, 99) if proc.size else 1.0
    proc_norm = (proc / max(p99,1.0)).clip(0,1)

    results = []

    # Experiment 1: baseline
    print('\nExperiment: baseline')
    base_rmse, base_node = create_windows_and_train(X, A, epochs=25)
    results.append({'exp':'baseline','overall_rmse':float(base_rmse)})

    # Experiment 2: +procedimentos
    print('\nExperiment: +procedimentos')
    X_p = np.concatenate([X, proc_norm[:,:,None]], axis=2)
    p_rmse, p_node = create_windows_and_train(X_p, A, epochs=25)
    results.append({'exp':'procedimentos','overall_rmse':float(p_rmse)})

    # Experiment 3: +procedimentos + faccao
    print('\nExperiment: +procedimentos + faccao')
    # expand faction onehot over time (static per-node)
    F_one = faction_onehot[np.newaxis, :, :].repeat(X.shape[0], axis=0)  # (T,N,Ff)
    X_pf = np.concatenate([X_p, F_one], axis=2)
    pf_rmse, pf_node = create_windows_and_train(X_pf, A, epochs=25)
    results.append({'exp':'procedimentos_faccao','overall_rmse':float(pf_rmse)})

    # Experiment 4: +procedimentos + faccao + exogenos
    print('\nExperiment: +procedimentos + faccao + exogenos')
    exog_stack = np.stack([exog['homic'], exog['armas'], exog['drogas'], exog['dinheiro']], axis=2)
    # normalize each channel globally
    for k in range(exog_stack.shape[2]):
        p99 = np.percentile(exog_stack[:,:,k], 99)
        if p99 <= 0: p99 = 1.0
        exog_stack[:,:,k] = (exog_stack[:,:,k] / p99).clip(0,1)
    X_full = np.concatenate([X_pf, exog_stack], axis=2)
    full_rmse, full_node = create_windows_and_train(X_full, A, epochs=25)
    results.append({'exp':'procedimentos_faccao_exogenos','overall_rmse':float(full_rmse)})

    # Save summary
    dfres = pd.DataFrame(results)
    dfres.to_csv('outputs/exog_faction_summary.csv', index=False)

    # Save per-bairro RMSE comparison (use last available per-node arrays)
    df_nodes = pd.DataFrame({'bairro_id': list(range(len(base_node))),
                             'rmse_baseline': base_node,
                             'rmse_proc': p_node,
                             'rmse_proc_facc': pf_node,
                             'rmse_full': full_node})
    df_nodes['improve_proc'] = df_nodes['rmse_baseline'] - df_nodes['rmse_proc']
    df_nodes['improve_full'] = df_nodes['rmse_baseline'] - df_nodes['rmse_full']
    df_nodes.to_csv('outputs/exog_faction_per_bairro_rmse.csv', index=False)

    # focused report for selected bairro
    fokus = df_nodes[df_nodes['bairro_id'] == target_bairro].copy()
    if not fokus.empty:
        fokus['bairro_name'] = target_name
        fokus.to_csv('outputs/exog_faction_focused_bairro.csv', index=False)
        print(f"Focused bairro report written to outputs/exog_faction_focused_bairro.csv for id={target_bairro}")
    else:
        print('Warning: selected bairro id not present in per-bairro RMSE table')

    # --- Focused run: subset to selected bairro and train with a 1-year window (365 days) for 50 epochs
    try:
        print('\nFocused run: bairro id=', target_bairro, ', janela=365 dias, epochs=50')
        # subset base and full tensors to the target bairro
        X_base_b = X[:, target_bairro:target_bairro+1, :]
        X_full_b = X_full[:, target_bairro:target_bairro+1, :]
        A_b = np.ones((1, 1), dtype=float)

        # train baseline (only original features for this node)
        brmse, brmse_node = create_windows_and_train(X_base_b, A_b, epochs=50, window_size=365)
        frmse, frn_node = create_windows_and_train(X_full_b, A_b, epochs=50, window_size=365)

        focused = {
            'bairro_id': int(target_bairro),
            'bairro_name': target_name,
            'baseline_rmse_1yr50ep': float(brmse),
            'full_rmse_1yr50ep': float(frmse),
            'delta_full_minus_baseline': float(brmse - frmse)
        }
        pd.DataFrame([focused]).to_csv('outputs/exog_faction_focused_bairro_detailed.csv', index=False)
        print('Focused detailed results saved to outputs/exog_faction_focused_bairro_detailed.csv')
    except Exception as e:
        print('Focused run failed:', e)

    print('\nExperiments complete. Summary written to outputs/exog_faction_summary.csv')


if __name__ == '__main__':
    main()
