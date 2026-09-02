Script nào sinh ra bảng nào:
  Bảng 1, 3  -> main_clean.py       (lambda = 0.4)
  Bảng 3     -> main_clean_rank00.py (lambda = 0)
  Bảng 2     -> main_heldout_drug.py / main_heldout_se.py
  Bảng 4     -> case_study.py

Seed: 1 (random, numpy, torch)
Loss: MSE + pairwise ranking, lambda cố định 0.4, margin 0.1
Các tùy chọn Huber, top-k, top-label weighting, prediction sharpening,
error-focus weighting: TẮT trong mọi kết quả báo cáo.
Không dùng ranking-weight scheduling, gap-based pair selection, hard-pair mining.

main_earlier_config.py là cấu hình dùng cho bản nộp ban đầu, giữ lại để đối chiếu.

Môi trường: transformers 4.35.2, huggingface_hub 0.19.4, tokenizers 0.15.2, subword-nmt
