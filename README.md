# Dual-BERT for Drug Side-Effect Frequency Prediction

Code accompanying the ICTA 2026 paper *"Dual-BERT for Drug Side-Effect
Frequency Prediction: An Empirical Comparison with a Substructure Transformer"*.

## Which script produces which table

| Table | Script |
|---|---|
| Table 1 (main comparison) | `main_clean.py` |
| Table 2 (entity held-out) | `main_heldout_drug.py`, `main_heldout_se.py` |
| Table 3 (ranking-loss ablation) | `main_clean.py` (λ=0.4) and `main_clean_rank00.py` (λ=0) |
| Table 4 (case study) | `case_study.py` |

Metrics are computed with the corresponding `compute_metrics_*.py` scripts.

## Training configuration used for all reported results

- Random seed: 1, set for `random`, `numpy` and `torch`
- Loss: MSE + pairwise ranking loss, fixed λ = 0.4, margin m = 0.1
- No ranking-weight scheduling, no gap-based pair selection, no hard-pair mining
- Huber loss, top-k optimisation, top-label weighting, prediction sharpening and
  error-focused weighting are present as options but are **disabled** in every
  reported run (`loss_type: mse`, corresponding weights 0 or neutral)

`main_earlier_config.py` is the configuration used for the originally submitted
version, kept for reference. It is **not** the source of any number in the
revised paper. Outputs from that configuration are kept in `results_old_config/`.

## Data

The benchmark data (750 drugs, 994 side effects, 37,071 annotated pairs) comes
from the HSTrans study: https://github.com/Dtdtxuky/HSTrans

`data/drug_codes_chembl_freq_1500.txt` and
`data/subword_units_map_chembl_freq_1500.csv` are the ESPF substructure
vocabulary of MolTrans (2,586 tokens, ChEMBL corpus, minimum frequency 1,500).

`data/sub/` holds the per-fold side-effect substructure representations; the fold
suffix reflects that they are extracted from the training partition of each fold only.

## Environment

transformers 4.35.2 · huggingface_hub 0.19.4 · tokenizers 0.15.2 · subword-nmt
