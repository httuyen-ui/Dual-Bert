import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import BertConfig, BertModel
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu gói transformers. Cài: pip install transformers"
    ) from exc

import torch
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE


def _attention_heads_for_hidden(hidden_size):
    for nh in (12, 8, 6, 5, 4, 3, 2):
        if hidden_size % nh == 0:
            return nh
    return 1


def _align_seq_len(x, target_len):
    """[B, L, D] -> cắt hoặc pad về L == target_len."""
    b, t, d = x.shape
    if t == target_len:
        return x
    if t > target_len:
        return x[:, :target_len, :].contiguous()
    pad = target_len - t
    return F.pad(x, (0, 0, 0, pad))


def _cnn_flat_dim(max_seq_len):
    """Hai Conv2d 3x3 không pad: chiều không gian L -> L-2 -> L-4."""
    h = max_seq_len - 4
    return 32 * h * h


def _bert_config(vocab_size, hidden_size, num_layers, max_seq_len):
    """BertModel encoder: cùng pattern cho nhánh thuốc và nhánh SE."""
    nh = _attention_heads_for_hidden(hidden_size)
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=nh,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_seq_len,
        type_vocab_size=2,
        pad_token_id=0,
    )


class Trans(nn.Module):
    """
    Hai nhánh BertModel (train từ config, không dùng pretrained HF cho SMILES):
      - Thuốc: id BPE SMILES + mask.
      - SE: id cấu trúc con + mask.
    Tương tác scaled dot -> CNN -> MLP -> một scalar.
    """

    def __init__(
        self,
        drug_num_layers=4,
        se_num_layers=8,
        max_seq_len=50,
        drug_hidden=300,
        drug_vocab_size=2586,
        se_vocab_size=2586,
        use_cross_attention=False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = drug_hidden
        self._interaction_dim = _cnn_flat_dim(max_seq_len)

        cfg_d = _bert_config(drug_vocab_size, drug_hidden, drug_num_layers, max_seq_len)
        cfg_s = _bert_config(se_vocab_size, drug_hidden, se_num_layers, max_seq_len)
        self.drug_bert = BertModel(cfg_d)
        self.se_bert = BertModel(cfg_s)

        # Dropout trên i_v: quá cao → nhiễu ranking / overlap; 0.1 ổn định hơn.
        self.dropout = 0.1
        self.icnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self._interaction_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
        self.CrossAttention = bool(use_cross_attention)
        if self.CrossAttention:
            num_heads = _attention_heads_for_hidden(drug_hidden)
            self.cross_attn_drug = nn.MultiheadAttention(
                embed_dim=drug_hidden,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.cross_attn_se = nn.MultiheadAttention(
                embed_dim=drug_hidden,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.cross_attn_norm_drug = nn.LayerNorm(drug_hidden)
            self.cross_attn_norm_se = nn.LayerNorm(drug_hidden)

    def set_bert_frozen(self, frozen: bool):
        """True: chỉ đóng băng drug_bert; se_bert + CNN + decoder vẫn train."""
        for p in self.drug_bert.parameters():
            p.requires_grad = not frozen

    def forward(self, Drug, SE, DrugMask, SEMsak):
        device = next(self.parameters()).device
        b = Drug.size(0)

        drug_ids = Drug.long().to(device)
        drug_m = DrugMask.long().to(device)
        x_d = self.drug_bert(input_ids=drug_ids, attention_mask=drug_m).last_hidden_state

        se_ids = SE.long().to(device)
        se_m = SEMsak.long().to(device)
        x_e = self.se_bert(input_ids=se_ids, attention_mask=se_m).last_hidden_state

        x_d = _align_seq_len(x_d, self.max_seq_len)
        x_e = _align_seq_len(x_e, self.max_seq_len)

        if self.CrossAttention:
            drug_key_padding_mask = (drug_m == 0)
            se_key_padding_mask = (se_m == 0)
            x_d_cross, _ = self.cross_attn_drug(
                query=x_d,
                key=x_e,
                value=x_e,
                key_padding_mask=se_key_padding_mask,
                need_weights=False,
            )
            x_e_cross, _ = self.cross_attn_se(
                query=x_e,
                key=x_d,
                value=x_d,
                key_padding_mask=drug_key_padding_mask,
                need_weights=False,
            )
            x_d = self.cross_attn_norm_drug(x_d + x_d_cross)
            x_e = self.cross_attn_norm_se(x_e + x_e_cross)

        scale = self.hidden_size ** 0.5
        i_v = torch.matmul(x_d, x_e.transpose(1, 2)) / scale
        i_v = i_v.unsqueeze(1)
        i_v = F.dropout(i_v, p=self.dropout)

        f = self.icnn(i_v).view(b, self._interaction_dim)
        score = self.decoder(f)
        return score, Drug, SE


def drug2emb_encoder(smile, max_len=50):
    """SMILES -> vector chỉ số BPE + mask (Dataset / identify_sub)."""
    vocab_path = "data/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("data/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator="")
    idx2word_d = sub_csv["index"].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = max_len
    t1 = dbpe.process_line(smile).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except Exception:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), "constant", constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
