# -*- coding: utf-8 -*-
"""
main_heldout_se.py — giao thuc GIU LAI THEO TAC DUNG PHU (GroupKFold tren ma SE),
cot "Side-effect-held-out" cua Bang 2 trong bai ICTA 2026 "Dual-BERT".
Muc tieu huan luyen: MSE + ranking loss, lambda = 0.4, margin = 0.1.

    python main_heldout_se.py

Cac tuy chon khac co trong ma nguon deu TAT trong moi ket qua bao cao:
    loss_type = mse | rank_gap_q = 0.0 | hard_pair_topk_ratio = 1.0, hard_pair_factor = 1
    topk_loss_weight = 0.0 | top_focus_weight = 1.0 | pred_sharp_scale = 1.0
    error_focus_q = 0.0 | focus_schedule = none | rank_norm_batch = False

Sieu tham so: Adam, lr = 5e-5, batch_size = 64, 40 epoch, weight_decay = 0.01. SEED = 1.
Cac default duoi day DA dat dung bang cau hinh da chay, nen chay khong tham so se tai lap
dung so trong bai. Mot phien ban truoc dat default khac (loss_type=mix, topk_loss_weight=0.2,
pred_sharp_scale=1.5, top_focus_weight=3.0, rank_gap_q=0.5, hard_pair_topk_ratio=0.3,
hard_pair_factor=4, lr=1e-4, batch_size=32, epoch=200); cau hinh do KHONG sinh ra so nao
trong bai.
"""
import argparse
import os
import pickle
import scipy
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import io
from Net import *
from smiles2vector import load_drug_smile
from math import *
import random
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
from sklearn.metrics import precision_score, recall_score, accuracy_score
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109
gii = open('data/drug_side.pkl', 'rb')
drug_side = pickle.load(gii)
gii.close()

# ===== [CLEAN CONFIG - Path A] =====
import torch as _torch_seed
SEED = 1
random.seed(SEED); np.random.seed(SEED)
_torch_seed.manual_seed(SEED); _torch_seed.cuda.manual_seed_all(SEED)
# Thu muc luu prediction rieng cho lan chay sach (khong de len .npy cu)
PREDICT_DIR = 'predictResult_heldout_se'
# ===================================


def log_print(message, log_path=None):
    text = message if isinstance(message, str) else str(message)
    print(text)
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


def result_print(message, result_path=None):
    """Ghi kết quả (tóm tắt / metric) vào file, append."""
    if not result_path:
        return
    text = message if isinstance(message, str) else str(message)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def Extract_positive_negative_samples(DAL, addition_negative_number=''):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # Sắp xếp hàng theo cột cuối
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def pairwise_ranking_loss(
    pred,
    label,
    margin,
    max_pairs,
    device,
    rank_norm_batch=False,
    hard_pair_factor=4,
    hard_pair_topk_ratio=0.5,
    rank_gap_q=0.0,
):
    """
    Cặp (i,j) với label_i > label_j: khuyến khích pred_i > pred_j + margin (hinge).
    Lấy mẫu ngẫu nhiên các cặp chỉ số trong batch để giới hạn chi phí.
    rank_norm_batch: chuẩn hóa pred trong batch (mean/std) chỉ cho nhánh ranking — giúp tối ưu thứ tự.
    rank_gap_q > 0: ưu tiên cặp có khoảng cách label (li-lj) lớn (>= phân vị q) hoặc cặp sai thứ tự
    (pred chưa đạt margin so với label).
    """
    pred = pred.flatten()
    if rank_norm_batch and pred.size(0) > 1:
        pred = pred - pred.mean()
        pred = pred / (pred.std() + 1e-8)
    label = label.flatten().float()
    n = pred.size(0)
    if n < 2:
        return pred.new_zeros(())
    n_sample = min(max_pairs * max(1, int(hard_pair_factor)), max(n * (n - 1), 1))
    i = torch.randint(0, n, (n_sample,), device=device)
    j = torch.randint(0, n, (n_sample,), device=device)
    m = i != j
    i, j = i[m], j[m]
    if i.numel() == 0:
        return pred.new_zeros(())
    li, lj = label[i], label[j]
    pi, pj = pred[i], pred[j]
    valid = li > lj
    if not valid.any():
        return pred.new_zeros(())
    li, lj, pi, pj = li[valid], lj[valid], pi[valid], pj[valid]
    gap = li - lj
    misordered = (pi - pj) < margin
    if rank_gap_q > 0.0 and gap.numel() > 0:
        q = min(max(float(rank_gap_q), 1e-5), 1.0 - 1e-5)
        try:
            thr = torch.quantile(gap.detach(), q)
            sel = (gap >= thr) | misordered
            if sel.any():
                li, lj, pi, pj = li[sel], lj[sel], pi[sel], pj[sel]
        except Exception:
            pass
    hinge = torch.relu(margin - (pi - pj))
    if hinge.numel() == 0:
        return pred.new_zeros(())
    if hard_pair_topk_ratio < 1.0:
        k = max(1, int(hinge.numel() * max(0.0, hard_pair_topk_ratio)))
        hinge = torch.topk(hinge, k=k, largest=True).values
    return hinge.mean()


def _apply_pred_sharpening(pred, sharp_scale, sharp_tanh):
    """Chỉnh pred trước ranking/top-k: scale hoặc tanh (tùy chọn)."""
    if sharp_tanh:
        return torch.tanh(pred * 2.0)
    if sharp_scale is not None and float(sharp_scale) != 1.0:
        return pred * float(sharp_scale)
    return pred


def batch_topk_overlap_loss(pred, label, k, topk_temp=0.2):
    """
    Differentiable top-k proxy:
    - target: mask top-k theo label (detach)
    - pred: softmax(pred / temperature)
    - loss: 1 - mass(pred) rơi vào target-topk
    Cách này có gradient ổn định hơn so với giao rời rạc bằng chỉ số topk.
    """
    pred = pred.flatten()
    label = label.flatten().float()
    n = pred.numel()
    if k is None or int(k) <= 0 or n < 2:
        return pred.new_zeros(())
    kk = min(int(k), int(n))
    _, idx_t = torch.topk(label, kk, largest=True)
    idx_t = idx_t.detach()
    vt = pred.new_zeros(n)
    vt.scatter_(0, idx_t, 1.0)
    vt = vt / float(kk)
    tau = max(float(topk_temp), 1e-4)
    p = torch.softmax(pred / tau, dim=0)
    topk_mass = (p * vt).sum()
    return 1.0 - topk_mass


def training_loss(
    pred,
    label,
    device,
    mse_weight=1.0,
    rank_weight=0.0,
    rank_margin=0.1,
    max_rank_pairs=512,
    rank_norm_batch=False,
    loss_type='mix',
    huber_delta=1.0,
    top_focus_ratio=0.2,
    top_focus_weight=2.0,
    hard_pair_factor=4,
    hard_pair_topk_ratio=0.5,
    top_focus_mode='quantile',
    error_focus_q=0.0,
    error_focus_mult=4.0,
    focus_schedule='none',
    focus_schedule_k=3.0,
    epoch=1,
    num_epoch=1,
    rank_gap_q=0.0,
    topk_loss_weight=0.0,
    topk_k=10,
    topk_temp=0.2,
    pred_sharp_scale=1.0,
    pred_sharp_tanh=False,
):
    """
    Tổng: mse_weight * reg + rank_weight * ranking + topk_loss_weight * topk_overlap.
    reg: MSE / Huber / mix + trọng số mẫu (top label, error focus, schedule).
    Ranking & top-k dùng pred đã sharpen (nếu bật); reg dùng pred gốc để không lệch thang đo.
    """
    pred = pred.flatten().to(device)
    label = label.flatten().float().to(device)
    pred_ord = _apply_pred_sharpening(pred, pred_sharp_scale, pred_sharp_tanh)

    err = pred - label
    mse_each = err ** 2
    if huber_delta <= 0:
        huber_each = torch.abs(err)
    else:
        ae = torch.abs(err)
        huber_each = torch.where(
            ae < huber_delta,
            0.5 * (ae ** 2) / huber_delta,
            ae - 0.5 * huber_delta,
        )
    if loss_type == 'mse':
        reg_each = mse_each
    elif loss_type == 'huber':
        reg_each = huber_each
    else:
        reg_each = 0.5 * mse_each + 0.5 * huber_each

    n = pred.size(0)
    w = torch.ones_like(reg_each, dtype=reg_each.dtype, device=reg_each.device)

    if n > 1 and error_focus_q > 0.0 and error_focus_mult > 1.0:
        ae = torch.abs(pred - label)
        try:
            qe = min(max(float(error_focus_q), 1e-5), 1.0 - 1e-5)
            err_thr = torch.quantile(ae.detach(), qe)
            w = torch.where(ae >= err_thr, torch.full_like(w, float(error_focus_mult)), w)
        except Exception:
            pass

    if n > 1 and top_focus_ratio > 0 and top_focus_weight > 1.0:
        try:
            if str(top_focus_mode).lower() == 'topk':
                k = max(1, int(float(top_focus_ratio) * n))
                k = min(k, n)
                top_idx = torch.topk(label, k=k, largest=True).indices
                w[top_idx] = w[top_idx] * float(top_focus_weight)
            else:
                q = min(max(1.0 - float(top_focus_ratio), 0.0), 1.0)
                threshold = torch.quantile(label.detach(), q)
                w = torch.where(label >= threshold, w * float(top_focus_weight), w)
        except Exception:
            pass

    if focus_schedule and str(focus_schedule).lower() != 'none' and num_epoch > 0:
        t = float(epoch) / float(max(num_epoch, 1))
        if str(focus_schedule).lower() == 'linear':
            w = 1.0 + (w - 1.0) * t
        elif str(focus_schedule).lower() == 'scaled':
            factor = 1.0 + t * float(focus_schedule_k)
            w = 1.0 + (w - 1.0) * factor

    reg = (reg_each * w).mean()

    total = mse_weight * reg

    if rank_weight > 0:
        r = pairwise_ranking_loss(
            pred_ord,
            label,
            rank_margin,
            max_rank_pairs,
            device,
            rank_norm_batch=rank_norm_batch,
            hard_pair_factor=hard_pair_factor,
            hard_pair_topk_ratio=hard_pair_topk_ratio,
            rank_gap_q=rank_gap_q,
        )
        total = total + rank_weight * r

    if topk_loss_weight > 0 and topk_k > 0 and n >= 2:
        tk = batch_topk_overlap_loss(pred_ord, label, topk_k, topk_temp=topk_temp)
        total = total + float(topk_loss_weight) * tk

    return total


def identify_sub(data, sub_run_id, drug_max_len=50):
    print('Đang trích xuất cấu trúc con hợp lệ')
    drug_smile = [item[1] for item in data]
    side_id = [item[0] for item in data]
    labels = [item[2] for item in data]

    # Lấy chỉ số SMILE–sub
    sub_dict = {}
    for i in range(len(drug_smile)):
        drug_sub, mask = drug2emb_encoder(drug_smile[i], max_len=drug_max_len)
        drug_sub = drug_sub.tolist()
        sub_dict[i] = drug_sub

    # Lưu tạm ra file
    with open(f'data/sub/my_dict_{sub_run_id}.pkl', 'wb') as f:
        pickle.dump(sub_dict, f)
    # Đọc lại file
    with open(f'data/sub/my_dict_{sub_run_id}.pkl', 'rb') as f:
        sub_dict = pickle.load(f)

    SE_sub = np.zeros((994, 2686))
    for j in range(len(drug_smile)):
        sideID = side_id[j]
        label = float(labels[j])
        for sub_idx in sub_dict[j]:
            if sub_idx == 0:
                continue
            SE_sub[int(sideID)][int(sub_idx)] += label

    np.save(f"data/sub/SE_sub_{sub_run_id}.npy", SE_sub)
    SE_sub = np.load(f"data/sub/SE_sub_{sub_run_id}.npy", allow_pickle=True)

    # Tổng toàn ma trận
    n = np.sum(SE_sub)
    # Tổng theo hàng
    SE_sum = np.sum(SE_sub, axis=1)
    SE_p = SE_sum / n
    # Tổng theo cột
    Sub_sum = np.sum(SE_sub, axis=0)
    Sub_p = Sub_sum / n

    SE_sub_p = SE_sub / n

    freq = np.zeros((994, 2686))
    for i in range(994):
        print(i)
        for j in range(2686):
            freq[i][j] = ((SE_sub_p[i][j] - SE_p[i] * Sub_p[j]) / (sqrt((SE_p[i] * Sub_p[j] / n)
                                                                        * (1 - SE_p[i]) *
                                                                        (1 - Sub_p[j])))) + 1e-5
    np.save(f"data/sub/freq_{sub_run_id}.npy", freq)
    freq = np.load(f"data/sub/freq_{sub_run_id}.npy", allow_pickle=True)
    non_nan_values = freq[~np.isnan(freq)]
    percentile_95 = np.percentile(non_nan_values, 95)
    print("Phân vị 95%:", percentile_95)

    l = []
    SE_sub_index = np.zeros((994, 50))
    for i in range(994):
        col = 0
        sorted_indices = np.argsort(freq[i])[::-1]
        filtered_indices = sorted_indices[freq[i][sorted_indices] > percentile_95]
        l.append(len(filtered_indices))
        for j in filtered_indices:
            if col < 50:
                SE_sub_index[i][col] = j
                col = col + 1
            else:
                continue

    np.save(f"data/sub/SE_sub_index_50_{sub_run_id}.npy", SE_sub_index)
    SE_sub_index = np.load(f"data/sub/SE_sub_index_50_{sub_run_id}.npy")

    SE_sub_mask = SE_sub_index.copy()
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save(f"data/sub/SE_sub_mask_50_{sub_run_id}.npy", SE_sub_mask)
    np.save("len_sub", l)


def trainfun(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    num_epoch,
    log_interval,
    test_loader,
    log_path=None,
    mse_weight=1.0,
    rank_weight=0.0,
    rank_margin=0.1,
    max_rank_pairs=512,
    rank_norm_batch=False,
    loss_type='mix',
    huber_delta=1.0,
    top_focus_ratio=0.2,
    top_focus_weight=2.0,
    hard_pair_factor=4,
    hard_pair_topk_ratio=0.5,
    top_focus_mode='quantile',
    error_focus_q=0.0,
    error_focus_mult=4.0,
    focus_schedule='none',
    focus_schedule_k=3.0,
    rank_gap_q=0.0,
    topk_loss_weight=0.0,
    topk_k=10,
    topk_temp=0.2,
    pred_sharp_scale=1.0,
    pred_sharp_tanh=False,
    freeze_bert_epochs=0,
    use_amp=False,
    scaler=None,
):
    # Bật chế độ huấn luyện
    model.train()
    if hasattr(model, 'set_bert_frozen'):
        model.set_bert_frozen(freeze_bert_epochs > 0 and epoch <= freeze_bert_epochs)

    batch_objectives = []
    amp_on = use_amp and device.type == 'cuda' and scaler is not None

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(train_loader):
        Drug = Drug.to(device)
        SE = SE.to(device)
        DrugMask = DrugMask.to(device)
        SEMsak = SEMsak.to(device)
        Label = torch.as_tensor([int(item) for item in Label], dtype=torch.float32, device=device)
        if epoch == 1 and batch_idx == 0:
            log_print("Model: {}".format(next(model.parameters()).device), log_path)
            log_print("Input: {}".format(Drug.device), log_path)
            log_print("DrugMask: {}".format(DrugMask.device), log_path)
            log_print("Label: {}".format(Label.device), log_path)

        optimizer.zero_grad()
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)

        pred = out.to(device)

        if amp_on:
            from torch.amp import autocast
            with autocast("cuda"):
                loss = training_loss(
                    pred,
                    Label,
                    device,
                    mse_weight=mse_weight,
                    rank_weight=rank_weight,
                    rank_margin=rank_margin,
                    max_rank_pairs=max_rank_pairs,
                    rank_norm_batch=rank_norm_batch,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                    top_focus_ratio=top_focus_ratio,
                    top_focus_weight=top_focus_weight,
                    hard_pair_factor=hard_pair_factor,
                    hard_pair_topk_ratio=hard_pair_topk_ratio,
                    top_focus_mode=top_focus_mode,
                    error_focus_q=error_focus_q,
                    error_focus_mult=error_focus_mult,
                    focus_schedule=focus_schedule,
                    focus_schedule_k=focus_schedule_k,
                    epoch=epoch,
                    num_epoch=num_epoch,
                    rank_gap_q=rank_gap_q,
                    topk_loss_weight=topk_loss_weight,
                    topk_k=topk_k,
                    topk_temp=topk_temp,
                    pred_sharp_scale=pred_sharp_scale,
                    pred_sharp_tanh=pred_sharp_tanh,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = training_loss(
                pred,
                Label,
                device,
                mse_weight=mse_weight,
                rank_weight=rank_weight,
                rank_margin=rank_margin,
                max_rank_pairs=max_rank_pairs,
                rank_norm_batch=rank_norm_batch,
                loss_type=loss_type,
                huber_delta=huber_delta,
                top_focus_ratio=top_focus_ratio,
                top_focus_weight=top_focus_weight,
                hard_pair_factor=hard_pair_factor,
                hard_pair_topk_ratio=hard_pair_topk_ratio,
                top_focus_mode=top_focus_mode,
                error_focus_q=error_focus_q,
                error_focus_mult=error_focus_mult,
                focus_schedule=focus_schedule,
                focus_schedule_k=focus_schedule_k,
                epoch=epoch,
                num_epoch=num_epoch,
                rank_gap_q=rank_gap_q,
                topk_loss_weight=topk_loss_weight,
                topk_k=topk_k,
                topk_temp=topk_temp,
                pred_sharp_scale=pred_sharp_scale,
                pred_sharp_tanh=pred_sharp_tanh,
            )
            loss.backward()
            optimizer.step()

        batch_objectives.append(loss.detach().item())

        if log_interval and batch_idx % log_interval == 0:
            with torch.no_grad():
                mse_mean = ((pred.flatten() - Label) ** 2).mean().item()
            log_print(
                'Train epoch: {} ({:.0f}%)\tMSE: {:.6e}\ttotal loss: {:.6e}'.format(
                    epoch,
                    100. * (batch_idx + 1) / len(train_loader),
                    mse_mean,
                    loss.item(),
                ),
                log_path,
            )

    return sum(batch_objectives) / len(batch_objectives) if batch_objectives else 0.0


def collect_all_predictions(model, device, test_loader):
    """All (label, pred) pairs in loader — for regression metrics and O@k%."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.as_tensor([int(item) for item in Label], dtype=torch.float32, device=device)
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)
            ys.append(Label.detach().cpu().numpy().flatten())
            ps.append(out.detach().cpu().numpy().flatten())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def format_epoch_line(epoch_1based, num_epoch, loss, rmse_v, mae_v, scc_v, o_metrics):
    return (
        'Epoch {:3d}/{} | Loss: {:.4f} | RMSE: {:.4f} | MAE: {:.4f} | SCC: {:.4f} | '
        'O@1%: {:.3f} | O@5%: {:.3f} | O@10%: {:.3f} | O@20%: {:.3f}'
    ).format(
        epoch_1based,
        num_epoch,
        loss,
        rmse_v,
        mae_v,
        scc_v,
        o_metrics[1],
        o_metrics[5],
        o_metrics[10],
        o_metrics[20],
    )


def predict(model, device, test_loader):
    # Gom kết quả vào tensor
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.as_tensor([int(item) for item in Label], dtype=torch.float32, device=device)
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)

            location = torch.where(Label != 0)
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, test_loader):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.as_tensor([int(item) for item in Label], dtype=torch.float32, device=device)
            output, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = output.detach().cpu()
            pred = torch.Tensor(pred)

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, Label.detach().cpu()), 0)

            pred = pred.numpy().flatten()
            pred = np.where(pred > 0.5, 1, 0)
            label = (Label.detach().cpu().numpy().flatten() != 0).astype(int)
            label = np.where(label != 0, 1, label)

            # Tránh warning khi batch chỉ có 1 lớp (all-0 hoặc all-1).
            if np.unique(label).size > 1:
                singleDrug_auc.append(roc_auc_score(label, pred))
                singleDrug_aupr.append(average_precision_score(label, pred))

        drugAUC = (sum(singleDrug_auc) / len(singleDrug_auc)) if singleDrug_auc else 0.0
        drugAUPR = (sum(singleDrug_aupr) / len(singleDrug_aupr)) if singleDrug_aupr else 0.0
        total_preds = total_preds.numpy()
        total_label = total_label.numpy()

        total_pre_binary = np.where(total_preds > 0.5, 1, 0)
        label01 = np.where(total_label != 0, 1, total_label)

        pre_list = total_pre_binary.tolist()
        label_list = label01.tolist()

        precision = precision_score(pre_list, label_list)

        # Recall
        recall = recall_score(pre_list, label_list)

        # Độ chính xác (accuracy)
        accuracy = accuracy_score(pre_list, label_list)

        total_preds = np.where(total_preds > 0.5, 1, 0)
        total_label = np.where(total_label != 0, 1, total_label)

        pos = np.squeeze(total_preds[np.where(total_label)])
        pos_label = np.ones(len(pos))

        neg = np.squeeze(total_preds[np.where(total_label == 0)])
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        if np.unique(y_true).size > 1:
            auc_all = roc_auc_score(y_true, y)
            aupr_all = average_precision_score(y_true, y)
        else:
            auc_all = 0.0
            aupr_all = 0.0

    return auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy


def main(
    training_generator,
    testing_generator,
    lr,
    num_epoch,
    weight_decay,
    log_interval,
    cuda_name,
    save_model,
    k,
    out_dir,
    checkpoint_every,
    trans_kwargs,
    mse_weight,
    rank_weight,
    rank_margin,
    max_rank_pairs,
    rank_norm_batch,
    loss_type,
    huber_delta,
    top_focus_ratio,
    top_focus_weight,
    hard_pair_factor,
    hard_pair_topk_ratio,
    top_focus_mode,
    error_focus_q,
    error_focus_mult,
    focus_schedule,
    focus_schedule_k,
    rank_gap_q,
    topk_loss_weight,
    topk_k,
    topk_temp,
    pred_sharp_scale,
    pred_sharp_tanh,
    freeze_bert_epochs,
    use_amp,
    log_file,
    result_file,
    best_model_metric,
    full_testing_generator=None,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(out_dir, log_file)
    result_path = os.path.join(out_dir, result_file)

    log_print('\n=======================================================================================', log_path)
    log_print('checkpoints: {}'.format(ckpt_dir), log_path)
    log_print('log file: {}'.format(log_path), log_path)
    log_print('result file: {}'.format(result_path), log_path)
    result_print('', result_path)
    result_print('=' * 80, result_path)
    result_print('fold: {} | time: {}'.format(k, datetime.now().isoformat(timespec='seconds')), result_path)
    result_print('checkpoints: {}'.format(ckpt_dir), result_path)
    result_print('=' * 80, result_path)

    log_print('time: ' + datetime.now().isoformat(timespec='seconds'), log_path)
    log_print('fold: {}'.format(k), log_path)
    log_print('model: ' + Trans.__name__, log_path)
    log_print('Learning rate: {}'.format(lr), log_path)
    log_print('Epochs: {}'.format(num_epoch), log_path)
    log_print('weight_decay: {}'.format(weight_decay), log_path)
    log_print('checkpoint_every: {}'.format(checkpoint_every), log_path)
    log_print(
        'mse_weight: {}\trank_weight: {}\trank_margin: {}\tmax_rank_pairs: {}\trank_norm_batch: {}\tloss_type: {}\thuber_delta: {}\ttop_focus_ratio: {}\ttop_focus_weight: {}\ttop_focus_mode: {}\terror_focus_q: {}\terror_focus_mult: {}\tfocus_schedule: {}\tfocus_schedule_k: {}\trank_gap_q: {}\ttopk_loss_weight: {}\ttopk_k: {}\ttopk_temp: {}\tpred_sharp_scale: {}\tpred_sharp_tanh: {}\thard_pair_factor: {}\thard_pair_topk_ratio: {}'.format(
            mse_weight, rank_weight, rank_margin, max_rank_pairs, rank_norm_batch,
            loss_type, huber_delta, top_focus_ratio, top_focus_weight, top_focus_mode,
            error_focus_q, error_focus_mult, focus_schedule, focus_schedule_k, rank_gap_q,
            topk_loss_weight, topk_k, topk_temp, pred_sharp_scale, pred_sharp_tanh,
            hard_pair_factor, hard_pair_topk_ratio),
        log_path,
    )
    log_print(
        'freeze_bert_epochs: {}\tuse_amp: {}\tTrans: {}'.format(
            freeze_bert_epochs, use_amp, trans_kwargs),
        log_path,
    )

    train_losses = []

    # Chọn thiết bị (CPU/GPU)
    log_print('CPU/GPU: {}'.format(torch.cuda.is_available()), log_path)
    if torch.cuda.is_available():
        try:
            device = torch.device(cuda_name)
            # Trigger kiểm tra hợp lệ device chỉ định (vd cuda:1 khi máy chỉ có 1 GPU).
            if device.type == 'cuda':
                _ = torch.cuda.get_device_name(device)
        except Exception:
            log_print(
                'Cảnh báo: cuda_name="{}" không hợp lệ, fallback về cuda:0'.format(cuda_name),
                log_path,
            )
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    log_print('Device: {}'.format(device), log_path)

    # Trans: hai BertModel (thuốc BPE + SE), train từ config
    model = Trans(**trans_kwargs).to(device)

    # Đếm tổng số tham số huấn luyện được
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print('Total parameters: {}'.format(total_params), log_path)

    # Bộ tối ưu (optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = None
    if use_amp and device.type == 'cuda':
        try:
            from torch.amp import GradScaler
            scaler = GradScaler("cuda")
        except Exception:
            scaler = None
            log_print('AMP: khong khoi tao duoc GradScaler, tat AMP.', log_path)

    # ----------------------------------------------------------------
    # RESUME — tu dong load checkpoint neu da chay truoc do
    # ----------------------------------------------------------------
    start_epoch = 0
    train_losses = []
    resume_path = os.path.join(ckpt_dir, 'fold{}_last.pt'.format(k))
    if os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            train_losses = ckpt.get('train_losses', [])
            log_print('Resume tu checkpoint: {} (da chay {} epoch)'.format(
                resume_path, start_epoch), log_path)
            if start_epoch >= num_epoch:
                log_print('Da du {} epoch, bo qua fold nay.'.format(num_epoch), log_path)
        except Exception as e:
            log_print('Khong load duoc: {} — bat dau tu dau.'.format(e), log_path)
            start_epoch = 0
            train_losses = []
    else:
        log_print('Khong co checkpoint — bat dau tu epoch 1.', log_path)

    best_overlap_score = -1.0
    best_balanced_score = -1.0
    for epoch in range(start_epoch, num_epoch):
        progress = float(epoch + 1) / float(max(num_epoch, 1))
        epoch_rank_weight = float(rank_weight)  # [CLEAN] constant lambda, no scheduling
        epoch_topk_weight = float(topk_loss_weight) * (0.2 + 0.8 * progress)
        train_loss = trainfun(
            model=model,
            device=device,
            train_loader=training_generator,
            optimizer=optimizer,
            epoch=epoch + 1,
            num_epoch=num_epoch,
            log_interval=log_interval,
            test_loader=testing_generator,
            log_path=log_path,
            mse_weight=mse_weight,
            rank_weight=epoch_rank_weight,
            rank_margin=rank_margin,
            max_rank_pairs=max_rank_pairs,
            rank_norm_batch=rank_norm_batch,
            loss_type=loss_type,
            huber_delta=huber_delta,
            top_focus_ratio=top_focus_ratio,
            top_focus_weight=top_focus_weight,
            hard_pair_factor=hard_pair_factor,
            hard_pair_topk_ratio=hard_pair_topk_ratio,
            top_focus_mode=top_focus_mode,
            error_focus_q=error_focus_q,
            error_focus_mult=error_focus_mult,
            focus_schedule=focus_schedule,
            focus_schedule_k=focus_schedule_k,
            rank_gap_q=rank_gap_q,
            topk_loss_weight=epoch_topk_weight,
            topk_k=topk_k,
            topk_temp=topk_temp,
            pred_sharp_scale=pred_sharp_scale,
            pred_sharp_tanh=pred_sharp_tanh,
            freeze_bert_epochs=freeze_bert_epochs,
            use_amp=use_amp and scaler is not None,
            scaler=scaler,
        )
        train_losses.append(train_loss)

        # RMSE/MAE/SCC chi tren positive (label>0) — Phuong an A
        y_t, y_p = collect_all_predictions(model, device, testing_generator)
        pos = y_t > 0
        rmse_v = rmse(y_t[pos], y_p[pos]) if pos.any() else 0.0
        mae_v  = MAE(y_t[pos],  y_p[pos]) if pos.any() else 0.0
        try:
            scc_v = float(spearman(y_t[pos], y_p[pos])) if pos.any() else 0.0
            if np.isnan(scc_v): scc_v = 0.0
        except Exception:
            scc_v = 0.0
        # Overlap tren toan bo T — Eq.20
        o_metrics = overlap_at_top_percent(y_t, y_p, percents=(1, 5, 10, 20))
        line = format_epoch_line(epoch + 1, num_epoch, train_loss, rmse_v, mae_v, scc_v, o_metrics)
        log_print(line, log_path)
        overlap_score = 0.6 * float(o_metrics[1]) + 0.3 * float(o_metrics[5]) + 0.1 * float(o_metrics[10])
        rmse_term = 1.0 / (1.0 + max(float(rmse_v), 0.0))
        mae_term = 1.0 / (1.0 + max(float(mae_v), 0.0))
        scc_term = max(float(scc_v), 0.0)
        balanced_score = (
            0.35 * float(o_metrics[1]) +
            0.20 * float(o_metrics[5]) +
            0.10 * float(o_metrics[10]) +
            0.20 * scc_term +
            0.10 * rmse_term +
            0.05 * mae_term
        )
        if save_model and overlap_score > best_overlap_score:
            best_overlap_score = overlap_score
            best_overlap_path = os.path.join(ckpt_dir, 'fold{}_best_overlap.pt'.format(k))
            torch.save({
                'fold': k,
                'epoch': epoch + 1,
                'best_overlap_score': best_overlap_score,
                'o_metrics': o_metrics,
                'rmse': rmse_v,
                'mae': mae_v,
                'scc': scc_v,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_losses': train_losses,
            }, best_overlap_path)
            log_print(
                'Saved best-overlap checkpoint: {} | score={:.4f} (O@1={:.3f}, O@5={:.3f}, O@10={:.3f})'.format(
                    best_overlap_path, best_overlap_score, o_metrics[1], o_metrics[5], o_metrics[10]
                ),
                log_path,
            )
        if save_model and balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_balanced_path = os.path.join(ckpt_dir, 'fold{}_best_balanced.pt'.format(k))
            torch.save({
                'fold': k,
                'epoch': epoch + 1,
                'best_balanced_score': best_balanced_score,
                'best_overlap_score': overlap_score,
                'o_metrics': o_metrics,
                'rmse': rmse_v,
                'mae': mae_v,
                'scc': scc_v,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_losses': train_losses,
            }, best_balanced_path)
            log_print(
                'Saved best-balanced checkpoint: {} | score={:.4f} (RMSE={:.4f}, MAE={:.4f}, SCC={:.4f}, O@1={:.3f})'.format(
                    best_balanced_path, best_balanced_score, rmse_v, mae_v, scc_v, o_metrics[1]
                ),
                log_path,
            )

        if save_model and checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0):
            ckpt_path = os.path.join(ckpt_dir, 'fold{}_epoch{}.pt'.format(k, epoch + 1))
            torch.save({
                'fold': k,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_losses': train_losses,
            }, ckpt_path)
            log_print('Saved checkpoint: {}'.format(ckpt_path), log_path)

    if save_model:
        last_path = os.path.join(ckpt_dir, 'fold{}_last.pt'.format(k))
        torch.save({
            'fold': k,
            'epoch': num_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1] if train_losses else None,
            'train_losses': train_losses,
        }, last_path)
        log_print('Saved last checkpoint: {}'.format(last_path), log_path)
        # Dung model epoch cuoi (last) de predict — giong HSTrans bao cao epoch 300
        log_print('Su dung model epoch cuoi de predict.', log_path)

    log_print('Dang du doan...', log_path)
    test_labels, test_preds = collect_all_predictions(
        model=model, device=device, test_loader=testing_generator)

    os.makedirs(PREDICT_DIR, exist_ok=True)
    np.save(f'{PREDICT_DIR}/total_labels_{k}.npy', test_labels)
    np.save(f'{PREDICT_DIR}/total_preds_{k}.npy', test_preds)

    # RMSE/MAE/SCC chi tren positive (label>0) — Phuong an A
    pos_test = test_labels > 0
    test_rMSE = rmse(test_labels[pos_test], test_preds[pos_test]) if pos_test.any() else 0.0
    test_MAE  = MAE(test_labels[pos_test],  test_preds[pos_test]) if pos_test.any() else 0.0
    try:
        test_spearman = float(spearman(test_labels[pos_test], test_preds[pos_test]))
        if np.isnan(test_spearman): test_spearman = 0.0
    except Exception:
        test_spearman = 0.0
    # Overlap@N% (Eq.20) tren PHAN BO THUC: test fold + toan bo negative con lai
    # RMSE/MAE/SCC van giu positive-only o tren
    if full_testing_generator is not None:
        full_labels, full_preds = collect_all_predictions(
            model=model, device=device, test_loader=full_testing_generator)
        o_metrics_test = overlap_at_top_percent(full_labels, full_preds, percents=(1, 5, 10, 20))
        log_print('[Overlap] Tinh tren phan bo thuc: T={} (pos={}, neg={})'.format(
            len(full_labels), int((full_labels > 0).sum()), int((full_labels == 0).sum())), log_path)
    else:
        o_metrics_test = overlap_at_top_percent(test_labels, test_preds, percents=(1, 5, 10, 20))
    loss_last = train_losses[-1] if train_losses else 0.0
    test_line = (
        'Test | Loss: {:.4f} | RMSE: {:.4f} | MAE: {:.4f} | SCC: {:.4f} | '
        'O@1%: {:.3f} | O@5%: {:.3f} | O@10%: {:.3f} | O@20%: {:.3f}'
    ).format(
        loss_last,
        test_rMSE,
        test_MAE,
        test_spearman,
        o_metrics_test[1],
        o_metrics_test[5],
        o_metrics_test[10],
        o_metrics_test[20],
    )
    log_print(test_line, log_path)

    result_print('Test (fold {}):'.format(k), result_path)
    result_print(test_line, result_path)
    result_print(
        'predict: predictResult/total_labels_{}.npy | predictResult/total_preds_{}.npy'.format(k, k),
        result_path,
    )
    result_print('-' * 80, result_path)


def _pad_to_len(arr, mask, target_len):
    arr = np.asarray(arr, dtype=np.int64).flatten()
    mask = np.asarray(mask, dtype=np.int64).flatten()
    if arr.size >= target_len:
        return arr[:target_len], mask[:target_len]
    pad_w = target_len - arr.size
    return (
        np.pad(arr, (0, pad_w), constant_values=0),
        np.pad(mask, (0, pad_w), constant_values=0),
    )


class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, fold_k, se_sub_version=32, max_seq_len=50):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.fold_k = fold_k
        self.se_sub_version = se_sub_version
        self.max_seq_len = max_seq_len
        # Cache SE substructure index/mask 1 lan (tranh np.load moi getitem -> nhanh hon rat nhieu)
        self._SE_index = np.load(
            f"data/sub/SE_sub_index_50_{se_sub_version}.npy"
        ).astype(int)
        self._SE_mask = np.load(f"data/sub/SE_sub_mask_50_{se_sub_version}.npy")
        # Cache ket qua ma hoa SMILES (chi 750 thuoc unique -> tinh 750 lan thay vi moi getitem)
        self._drug_cache = {}

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        d = self.df.iloc[index]['Drug_smile']
        s = int(self.df.iloc[index]['SE_id'])

        if d in self._drug_cache:
            arr, m = self._drug_cache[d]
        else:
            arr, m = drug2emb_encoder(d, max_len=self.max_seq_len)
            self._drug_cache[d] = (arr, m)
        d_v = torch.as_tensor(arr, dtype=torch.long)
        input_mask_d = torch.as_tensor(m, dtype=torch.long)

        s_v = self._SE_index[s, :]
        input_mask_s = self._SE_mask[s, :]
        s_v, input_mask_s = _pad_to_len(s_v, input_mask_s, self.max_seq_len)
        s_v = torch.as_tensor(s_v, dtype=torch.long)
        input_mask_s = torch.as_tensor(input_mask_s, dtype=torch.long)

        y = self.labels[index]
        return d_v, s_v, input_mask_d, input_mask_s, y


if __name__ == '__main__':
    # Tham số dòng lệnh
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=5e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.01, help='weight_decay')
    parser.add_argument('--epoch', type=int, required=False, default=40, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=0,
                        help='Print batch MSE every N batches (0 = only epoch summary)')
    parser.add_argument(
        '--cuda_name',
        type=str,
        required=False,
        default='cuda:0',
        help="Thiết bị chạy (vd: cuda:0, cuda:1, cpu). Mặc định ưu tiên GPU 0.",
    )
    parser.add_argument('--dim', type=int, required=False, default=200,
                        help='features dimensions of drugs and side effects')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='checkpoints',
        help='Thư mục output (mặc định: checkpoints).',
    )
    parser.add_argument('--log_file', type=str, default='log6.txt', help='Tên file log huấn luyện (trong out_dir)')
    parser.add_argument('--result_file', type=str, default='result6.txt', help='Tóm tắt metric test từng fold (append)')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help='save checkpoint every N epochs (0 = skip periodic saves; last.pt still saved if --save_model)')
    parser.add_argument(
        '--se_sub_version',
        type=int,
        default=32,
        help='Hậu tố SE_sub_index_50_{n}.npy / SE_sub_mask_50_{n}.npy và identify_sub(..., n). '
             'Đổi số nếu thư mục data/sub của bạn dùng hậu tố khác (ví dụ 0, 35).',
    )
    parser.add_argument(
        '--mse_weight',
        type=float,
        default=1.0,
        help='Trọng số MSE (ví dụ 0.3 khi kết hợp 0.3 MSE + 0.7 ranking).',
    )
    parser.add_argument(
        '--rank_weight',
        type=float,
        default=0.4,
        help='Trọng số ranking loss (0 = chỉ MSE). Thử 1.0 nếu SCC thấp.',
    )
    parser.add_argument(
        '--rank_margin',
        type=float,
        default=0.1,
        help='Margin hinge cho pairwise ranking (label cao hơn -> dự đoán cao hơn ít nhất margin).',
    )
    parser.add_argument(
        '--max_rank_pairs',
        type=int,
        default=512,
        help='Số cặp chỉ số ngẫu nhiên tối đa mỗi batch cho ranking loss.',
    )
    parser.add_argument(
        '--rank_norm_batch',
        action='store_true',
        help='Chuẩn hóa pred trong batch chỉ cho ranking loss (hỗ trợ thứ tự / SCC).',
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='mse',
        choices=['mse', 'huber', 'mix'],
        help='Loss hồi quy: mse, huber hoặc mix (0.5 mse + 0.5 huber).',
    )
    parser.add_argument(
        '--huber_delta',
        type=float,
        default=1.0,
        help='Delta cho Huber loss (chỉ dùng khi loss_type=huber/mix).',
    )
    parser.add_argument(
        '--top_focus_ratio',
        type=float,
        default=0.2,
        help='Tỉ lệ top label trong batch được tăng trọng số cho loss hồi quy.',
    )
    parser.add_argument(
        '--top_focus_weight',
        type=float,
        default=1.0,
        help='Trọng số cho nhóm top label trong batch.',
    )
    parser.add_argument(
        '--top_focus_mode',
        type=str,
        default='topk',
        choices=['quantile', 'topk'],
        help='quantile: như cũ (ngưỡng theo phân vị label). topk: torch.topk(label, k=ratio*n) rồi w[top]*=top_focus_weight — tốt cho overlap.',
    )
    parser.add_argument(
        '--error_focus_q',
        type=float,
        default=0.0,
        help='Phân vị trên |pred-label|; mẫu lỗi cao hơn ngưỡng nhận error_focus_mult (0 = tắt). Thử 0.8.',
    )
    parser.add_argument(
        '--error_focus_mult',
        type=float,
        default=4.0,
        help='Trọng số mẫu có |pred-label| >= phân vị error_focus_q.',
    )
    parser.add_argument(
        '--focus_schedule',
        type=str,
        default='none',
        choices=['none', 'linear', 'scaled'],
        help='none: không đổi theo epoch. linear: w=1+(w-1)*epoch/max. scaled: w=1+(w-1)*(1+epoch/max*k) với --focus_schedule_k.',
    )
    parser.add_argument(
        '--focus_schedule_k',
        type=float,
        default=3.0,
        help='Hệ số k khi focus_schedule=scaled (ví dụ báo: 1 + epoch/max * 3).',
    )
    parser.add_argument(
        '--rank_gap_q',
        type=float,
        default=0.0,
        help='Ranking: giữ cặp li>lj có (li-lj)>=phân vị q hoặc cặp sai thứ tự (pred chưa margin). 0=tắt. Thử 0.5.',
    )
    parser.add_argument(
        '--hard_pair_factor',
        type=int,
        default=1,
        help='Hệ số lấy thêm cặp ranking trước khi chọn hard pairs.',
    )
    parser.add_argument(
        '--hard_pair_topk_ratio',
        type=float,
        default=1.0,
        help='Tỉ lệ hard pairs có loss cao nhất dùng để tính ranking loss.',
    )
    parser.add_argument(
        '--topk_loss_weight',
        type=float,
        default=0.0,
        help='Top-k overlap trong batch: loss += weight * (1 - |topk(pred) ∩ topk(label)|/k). 0=tắt; thử 0.3.',
    )
    parser.add_argument(
        '--topk_temp',
        type=float,
        default=0.2,
        help='Nhiệt độ softmax cho top-k differentiable loss (nhỏ hơn -> tập trung top).',
    )
    parser.add_argument(
        '--topk_k',
        type=int,
        default=10,
        help='k cho top-k loss (cần batch_size >= k).',
    )
    parser.add_argument(
        '--pred_sharp_scale',
        type=float,
        default=1.0,
        help='Nhân pred trước ranking + top-k loss (1.0=tắt). Thử 1.5.',
    )
    parser.add_argument(
        '--pred_sharp_tanh',
        action='store_true',
        help='Dùng tanh(pred*2) cho pred vào ranking + top-k (ưu tiên hơn scale nếu cùng bật).',
    )
    parser.add_argument(
        '--best_model_metric',
        type=str,
        default='balanced',
        choices=['balanced', 'overlap'],
        help='Tiêu chí chọn checkpoint để predict cuối: balanced (giữ RMSE/MAE/SCC + overlap) hoặc overlap (ưu tiên O@k).',
    )
    parser.add_argument(
        '--drug_hidden',
        type=int,
        default=300,
        help='Chiều ẩn chung cho cả hai nhánh BERT (thuốc + SE).',
    )
    parser.add_argument(
        '--drug_bert_layers',
        type=int,
        default=4,
        help='Số tầng BertModel nhánh thuốc (SMILES BPE).',
    )
    parser.add_argument(
        '--bert_layers',
        type=int,
        default=8,
        help='Số tầng BertModel nhánh SE (id cấu trúc con).',
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=50,
        help='Độ dài chuỗi (thuốc + SE pad). Đổi (vd. 64) cần đồng bộ mask.',
    )
    parser.add_argument(
        '--freeze_bert_epochs',
        type=int,
        default=0,
        help='Số epoch đầu đóng băng drug_bert (se_bert + CNN + decoder vẫn train). 0 = tắt.',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Mixed precision (FP16) trên GPU — nhanh hơn khi có CUDA.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Kích thước batch.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Số worker DataLoader (thử 4/6/8 để tăng tốc nạp dữ liệu).',
    )
    parser.add_argument(
        '--pin_memory',
        action='store_true',
        help='Bật pin_memory cho DataLoader (nên bật khi train bằng CUDA).',
    )

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model
    out_dir = args.out_dir
    log_file = args.log_file
    result_file = args.result_file
    checkpoint_every = args.checkpoint_every
    se_sub_version = args.se_sub_version
    mse_weight = args.mse_weight
    rank_weight = args.rank_weight
    rank_margin = args.rank_margin
    max_rank_pairs = args.max_rank_pairs
    rank_norm_batch = args.rank_norm_batch
    loss_type = args.loss_type
    huber_delta = args.huber_delta
    top_focus_ratio = args.top_focus_ratio
    top_focus_weight = args.top_focus_weight
    top_focus_mode = args.top_focus_mode
    error_focus_q = args.error_focus_q
    error_focus_mult = args.error_focus_mult
    focus_schedule = args.focus_schedule
    focus_schedule_k = args.focus_schedule_k
    rank_gap_q = args.rank_gap_q
    topk_loss_weight = args.topk_loss_weight
    topk_k = args.topk_k
    topk_temp = args.topk_temp
    pred_sharp_scale = args.pred_sharp_scale
    pred_sharp_tanh = args.pred_sharp_tanh
    best_model_metric = args.best_model_metric
    hard_pair_factor = args.hard_pair_factor
    hard_pair_topk_ratio = args.hard_pair_topk_ratio
    trans_kwargs = {
        'drug_num_layers': args.drug_bert_layers,
        'se_num_layers': args.bert_layers,
        'max_seq_len': args.max_seq_len,
        'drug_hidden': args.drug_hidden,
    }
    freeze_bert_epochs = args.freeze_bert_epochs
    use_amp = args.amp
    batch_size = args.batch_size
    num_workers = max(0, int(args.num_workers))
    use_cuda_loader = torch.cuda.is_available() and str(cuda_name).lower() != 'cpu'
    pin_memory = bool(args.pin_memory or use_cuda_loader)

    # Lấy mẫu dương / âm
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    addition_negative_sample = np.vstack((addition_negative_sample, final_negative_sample))

    final_sample = final_positive_sample

    X = final_sample[:, 0::]

    final_target = final_sample[:, final_sample.shape[1] - 1]

    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    data_neg_smile = []   # (SE_id, Drug_smile, Label) — dung de tinh Overlap tren phan bo thuc
    drug_dict, drug_smile = load_drug_smile(SMILES_file)


    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append(
            (addition_negative_sample[i, 1], addition_negative_sample[i, 0], addition_negative_sample[i, 2]))
        data_neg_smile.append(
            (addition_negative_sample[i, 1], drug_smile[addition_negative_sample[i, 0]], addition_negative_sample[i, 2]))
    # DataFrame negative day du (~671k cap label=0) — KHONG dung de train, chi de tinh Overlap cuoi
    df_neg_full = pd.DataFrame(data=data_neg_smile, columns=['SE_id', 'Drug_smile', 'Label'])
    for i in range(X.shape[0]):
        data_x.append((X[i, 1], X[i, 0]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 1], drug_smile[X[i, 0]], X[i, 2]))

    # ===== [HELD-OUT: se-held-out] =====
    from sklearn.model_selection import GroupKFold
    _drug_ids = np.array([X[i, 0] for i in range(X.shape[0])])
    _se_ids   = np.array([X[i, 1] for i in range(X.shape[0])])
    _groups   = _se_ids
    print("[HELD-OUT] mode = se-held-out | so nhom =", len(np.unique(_groups)))
    # ===================================

    fold = 1
    kfold = StratifiedKFold(5, random_state=1, shuffle=True)

    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    test_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        train_params['persistent_workers'] = True
        test_params['persistent_workers'] = True
    print(
        "DataLoader config | batch_size={} num_workers={} pin_memory={} persistent_workers={}".format(
            batch_size,
            num_workers,
            pin_memory,
            num_workers > 0,
        )
    )

    # [CHONG LEAKAGE] identify_sub chay BEN TRONG fold, chi dung data_train
    _splitter = GroupKFold(n_splits=5).split(data_x, data_y, groups=_groups)
    for k, (train, test) in enumerate(_splitter):
        data_train = np.array(data)[train]
        data_test = np.array(data)[test]

        # Chi fit substructure tren data_train — tranh ro ri nhan test
        identify_sub(data_train.tolist(), k, drug_max_len=args.max_seq_len)

        # Chuyển sang DataFrame
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_test = pd.DataFrame(data=data_test.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])

        # Tạo Dataset và DataLoader
        # Dung k lam se_sub_version — moi fold doc file rieng SE_sub_index_50_{k}.npy
        training_set = Data_Encoder(
            df_train.index.values,
            df_train.Label.values,
            df_train,
            k,
            k,
            max_seq_len=args.max_seq_len,
        )
        testing_set = Data_Encoder(
            df_test.index.values,
            df_test.Label.values,
            df_test,
            k,
            k,
            max_seq_len=args.max_seq_len,
        )

        training_generator = torch.utils.data.DataLoader(training_set, **train_params)
        testing_generator = torch.utils.data.DataLoader(testing_set, **test_params)

        # Tap test phan bo thuc: test fold (50/50) + toan bo negative con lai (~671k)
        # Chi dung de tinh Overlap cuoi cung — KHONG train tren day -> khong leakage
        df_test_full = pd.concat([df_test, df_neg_full], ignore_index=True)
        full_testing_set = Data_Encoder(
            df_test_full.index.values,
            df_test_full.Label.values,
            df_test_full,
            k,
            k,
            max_seq_len=args.max_seq_len,
        )
        full_testing_generator = torch.utils.data.DataLoader(full_testing_set, **test_params)

        main(
            training_generator,
            testing_generator,
            lr,
            num_epoch,
            weight_decay,
            log_interval,
            cuda_name,
            save_model,
            k,
            out_dir,
            checkpoint_every,
            trans_kwargs,
            mse_weight,
            rank_weight,
            rank_margin,
            max_rank_pairs,
            rank_norm_batch,
            loss_type,
            huber_delta,
            top_focus_ratio,
            top_focus_weight,
            hard_pair_factor,
            hard_pair_topk_ratio,
            top_focus_mode,
            error_focus_q,
            error_focus_mult,
            focus_schedule,
            focus_schedule_k,
            rank_gap_q,
            topk_loss_weight,
            topk_k,
            topk_temp,
            pred_sharp_scale,
            pred_sharp_tanh,
            freeze_bert_epochs,
            use_amp,
            log_file,
            result_file,
            best_model_metric,
            full_testing_generator,
        )
