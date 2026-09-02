"""
Tinh metric THONG NHAT tu predictResult (cho ca Dual-BERT va HSTrans).
Doc total_labels_{k}.npy / total_preds_{k}.npy (toan bo T 50/50, da luu trong luc chay),
roi tinh:
  - RMSE/MAE/SCC: positive-only (loc label>0) — giong HSTrans
  - Overlap@N%: 50/50 (toan bo T, Eq.20 hits/k) — giong HSTrans
  - mean +/- std qua 5 fold

Dung:
  python compute_metrics.py            # Dual-BERT (prefix rong: total_labels_k.npy)
  python compute_metrics.py hstrans_   # HSTrans  (prefix 'hstrans_': hstrans_total_labels_k.npy)
"""
import sys
import numpy as np
from utils import rmse, MAE, spearman, overlap_at_top_percent

prefix = sys.argv[1] if len(sys.argv) > 1 else ''
n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

rmses, maes, sccs = [], [], []
o1s, o5s, o10s, o20s = [], [], [], []

print(f'=== Doc predictResult_heldout_se/{prefix}total_*_k.npy (k=0..{n_folds-1}) ===')
for k in range(n_folds):
    try:
        labels = np.load(f'predictResult_heldout_se/{prefix}total_labels_{k}.npy')
        preds = np.load(f'predictResult_heldout_se/{prefix}total_preds_{k}.npy')
    except FileNotFoundError:
        print(f'  [!] Thieu file fold {k} — bo qua')
        continue

    labels = np.asarray(labels, dtype=float).flatten()
    preds = np.asarray(preds, dtype=float).flatten()
    pos = labels > 0

    r = rmse(labels[pos], preds[pos])
    m = MAE(labels[pos], preds[pos])
    s = float(spearman(labels[pos], preds[pos]))
    o = overlap_at_top_percent(labels, preds, percents=(1, 5, 10, 20))

    rmses.append(r); maes.append(m); sccs.append(s)
    o1s.append(o[1]); o5s.append(o[5]); o10s.append(o[10]); o20s.append(o[20])

    print(f'  fold {k}: T={len(labels)} pos={int(pos.sum())} | '
          f'RMSE {r:.4f} MAE {m:.4f} SCC {s:.4f} | '
          f'O@1% {o[1]:.3f} O@5% {o[5]:.3f} O@10% {o[10]:.3f} O@20% {o[20]:.3f}')


def ms(x):
    return f'{np.mean(x):.4f} ± {np.std(x):.4f}' if x else 'N/A'


print('\n=== mean ± std (qua {} fold) ==='.format(len(rmses)))
print('RMSE :', ms(rmses))
print('MAE  :', ms(maes))
print('SCC  :', ms(sccs))
print('O@1% :', ms(o1s))
print('O@5% :', ms(o5s))
print('O@10%:', ms(o10s))
print('O@20%:', ms(o20s))
print('\nLuu y: RMSE/MAE/SCC = positive-only; Overlap = 50/50 (giong HSTrans Table 1).')
