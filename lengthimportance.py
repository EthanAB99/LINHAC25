# ==========================================================
# ev_analysis.py – EV Impact by Event Position (Using Trained Model)
# ==========================================================

import torch
import numpy as np
from torch.utils.data import DataLoader

from model_def import HockeyModel, HockeyDS, collate
from data_loader import X_test, y_test, players_test, event_vocab, type_vocab

# ---------- Load model ----------
model = HockeyModel(
    event_vocab_size=len(event_vocab) + 1,
    type_vocab_size=len(type_vocab) + 1
)

# Load and fix state dict
state_dict = torch.load("zone_exit_lstm_state_dict.pt")
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("lstm.module."):
        new_k = k.replace("lstm.module.", "lstm.")
    else:
        new_k = k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# ---------- Dataloader ----------
test_loader = DataLoader(
    HockeyDS(X_test, y_test),
    batch_size=32,
    shuffle=False,
    collate_fn=collate
)

# ---------- EV utility ----------
def expected_value(probs):
    controlled, uncontrolled, fail = probs
    return (controlled * 0.66) + (uncontrolled * 0.29)

# ---------- EV delta by position ----------
position_buckets = {i: [] for i in range(10)}

with torch.no_grad():
    for ev, tp, num, lbl, L, nums_full in test_loader:
        probs_seq = model.forward_sequence(ev, tp, num, L)

        for i in range(probs_seq.size(0)):
            probs = probs_seq[i]
            length = L[i]
            probs = probs[:length]

            if length < 2:
                continue

            for t in range(1, length):
                ev_before = expected_value(probs[t-1].cpu().numpy())
                ev_after = expected_value(probs[t].cpu().numpy())
                delta_ev = ev_after - ev_before

                bucket = min(int(t / (length - 1) * 10), 9)
                position_buckets[bucket].append(delta_ev)

# ---------- Calculate total and per-bucket shares ----------
total_ev_change = sum(sum(bucket) for bucket in position_buckets.values())

print("\n=== EV Change Percentage Breakdown by Sequence Position ===")
for bucket in range(10):
    bucket_sum = sum(position_buckets[bucket])
    if total_ev_change != 0:
        pct = (bucket_sum / total_ev_change) * 100
    else:
        pct = 0.0
    print(f"{bucket*10:2d}–{(bucket+1)*10:2d}% through sequence: {pct:+.2f}% of total EV change")

