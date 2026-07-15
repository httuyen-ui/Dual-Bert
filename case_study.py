"""
Case study: du doan 994 tac dung phu (SE) cho MOT thuoc (mac dinh Ibuprofen, index 199).
Khop dung voi main.py (Dual-BERT) cua Tuyen.

Chay (Dual-BERT 50M, fold 0):
    python case_study.py --ckpt checkpoint1/checkpoints/fold0_last.pt --fold 0

Chay cho ban small (25M):
    python case_study.py --ckpt checkpoint_small/checkpoints/fold0_last.pt --fold 0 \
        --drug_bert_layers 2 --bert_layers 4

Ket qua in ra man hinh + luu file --out.
"""
import argparse
import pickle
import numpy as np
import torch

from Net import *                      # Trans, drug2emb_encoder (giong main.py)
from smiles2vector import load_drug_smile


def main():
    parser = argparse.ArgumentParser(description='Case study: predict 994 SE for one drug')
    parser.add_argument('--ckpt', default='checkpoint1/checkpoints/fold0_last.pt',
                        help='Duong dan checkpoint (.pt)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Dung file SE_sub_index_50_{fold}.npy / SE_sub_mask_50_{fold}.npy')
    parser.add_argument('--drug_index', type=int, default=199, help='Index thuoc (Ibuprofen=199)')
    parser.add_argument('--drug_hidden', type=int, default=300)
    parser.add_argument('--drug_bert_layers', type=int, default=4)
    parser.add_argument('--bert_layers', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--cuda_name', default='cuda:0')
    parser.add_argument('--smiles_file', default='data/drug_SMILES_750.csv')
    parser.add_argument('--drug_side', default='data/drug_side.pkl')
    parser.add_argument('--se_names', default='',
                        help='(Tuy chon) file CSV/txt ten 994 SE, moi dong 1 ten. De trong neu khong co.')
    parser.add_argument('--out', default='case_study_ibuprofen.txt')
    args = parser.parse_args()

    device = torch.device(args.cuda_name if torch.cuda.is_available() else 'cpu')
    lines = []

    def emit(msg):
        print(msg)
        lines.append(str(msg))

    # ---------- 1. Nhan that (ground truth) ----------
    with open(args.drug_side, 'rb') as f:
        drug_side = pickle.load(f)             # (750, 994), label 0..5
    drug_side = np.asarray(drug_side)
    gt = drug_side[args.drug_index].astype(float)   # (994,)
    n_se = drug_side.shape[1]

    # ---------- 2. SMILES cua thuoc ----------
    drug_dict, drug_smile = load_drug_smile(args.smiles_file)
    smile = drug_smile[args.drug_index]
    emit('=' * 70)
    emit('CASE STUDY | drug_index = {} | checkpoint = {}'.format(args.drug_index, args.ckpt))
    emit('SMILES: {}'.format(smile))
    emit('So SE that (label>0): {}/{}'.format(int((gt > 0).sum()), n_se))
    emit('=' * 70)

    # ma hoa thuoc 1 lan
    d_arr, d_mask = drug2emb_encoder(smile, max_len=args.max_seq_len)
    d_arr = np.asarray(d_arr, dtype=np.int64).flatten()
    d_mask = np.asarray(d_mask, dtype=np.int64).flatten()

    # ---------- 3. SE substructure (theo fold) ----------
    SE_index = np.load('data/sub/SE_sub_index_50_{}.npy'.format(args.fold)).astype(int)  # (994,50)
    SE_mask = np.load('data/sub/SE_sub_mask_50_{}.npy'.format(args.fold))                 # (994,50)
    assert SE_index.shape[0] == n_se, 'So SE trong file sub khong khop drug_side'

    # batch: cung 1 thuoc lap lai cho moi SE
    Drug = torch.as_tensor(np.tile(d_arr, (n_se, 1)), dtype=torch.long)
    DrugMask = torch.as_tensor(np.tile(d_mask, (n_se, 1)), dtype=torch.long)
    SE = torch.as_tensor(SE_index, dtype=torch.long)
    SEMask = torch.as_tensor(SE_mask, dtype=torch.long)

    # ---------- 4. Load model ----------
    trans_kwargs = {
        'drug_num_layers': args.drug_bert_layers,
        'se_num_layers': args.bert_layers,
        'max_seq_len': args.max_seq_len,
        'drug_hidden': args.drug_hidden,
    }
    model = Trans(**trans_kwargs).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # ---------- 5. Du doan 994 SE ----------
    preds = []
    BS = 128
    with torch.no_grad():
        for i in range(0, n_se, BS):
            out, _, _ = model(
                Drug[i:i + BS].to(device),
                SE[i:i + BS].to(device),
                DrugMask[i:i + BS].to(device),
                SEMask[i:i + BS].to(device),
            )
            preds.append(out.detach().cpu().numpy().flatten())
    pred = np.concatenate(preds)[:n_se]

    # ---------- 6. Phan tich ----------
    se_names = None
    if args.se_names:
        try:
            with open(args.se_names, 'r', encoding='utf-8') as f:
                se_names = [ln.strip() for ln in f if ln.strip()]
            if len(se_names) < n_se:
                se_names = None
        except Exception:
            se_names = None

    order = np.argsort(pred)[::-1]
    emit('')
    emit('--- Top {} SE du doan tan suat cao nhat ---'.format(args.topk))
    header = 'rank | SE_id | pred  | true' + ('  | name' if se_names else '')
    emit(header)
    hit = 0
    for r, se in enumerate(order[:args.topk]):
        t = gt[se]
        if t > 0:
            hit += 1
        row = '{:4d} | {:5d} | {:.3f} | {:.0f}'.format(r + 1, int(se), pred[se], t)
        if se_names:
            row += '  | ' + se_names[se]
        emit(row)
    emit('')
    emit('Trong top-{}: {}/{} la SE that (label>0)'.format(args.topk, hit, args.topk))

    # Precision@k
    emit('')
    for k in (10, 20, 50):
        topk = order[:k]
        p = float((gt[topk] > 0).sum()) / k
        emit('Precision@{:<2d}: {:.3f}'.format(k, p))

    # SCC positive-only (giong metric chinh cua bai)
    try:
        from scipy.stats import spearmanr
        pos = gt > 0
        if pos.sum() > 1:
            scc, _ = spearmanr(pred[pos], gt[pos])
            emit('')
            emit('SCC (positive-only) cho thuoc nay: {:.3f}'.format(float(scc)))
    except Exception as e:
        emit('SCC loi: {}'.format(e))

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\nDa luu: {}'.format(args.out))


if __name__ == '__main__':
    main()
