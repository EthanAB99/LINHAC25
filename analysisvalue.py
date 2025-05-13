 # ==========================================================
# ev_analysis.py – EV Impact by Event Type & Location (No Outcome Tags)
# ==========================================================

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from model_def import HockeyModel, HockeyDS, collate
from data_loader import X_test, y_test, players_test, event_vocab, type_vocab

# Load model
model = HockeyModel(event_vocab_size=len(event_vocab)+1, type_vocab_size=len(type_vocab)+1)
model.load_state_dict(torch.load("zone_exit_lstm_state_dict.pt"), strict=False)
model.eval()

inv_event_vocab = {v: k for k, v in event_vocab.items()}
inv_type_vocab  = {v: k for k, v in type_vocab.items()}

test_loader = DataLoader(HockeyDS(X_test, y_test), batch_size=32, shuffle=False, collate_fn=collate)

def expected_value(probs):
    controlled, uncontrolled, fail = probs
    return (controlled * 0.38) + (uncontrolled * 0.16)+(fail*0.06)

MIN_SAMPLES = 30

# Containers
event_oteam_ev_deltas = defaultdict(list)
event_dteam_ev_deltas = defaultdict(list)
event_type_oteam_ev_deltas = defaultdict(list)
event_type_dteam_ev_deltas = defaultdict(list)
event_type_loc_oteam_deltas = defaultdict(list)
event_type_loc_dteam_ev_deltas = defaultdict(list)

with torch.no_grad():
    for batch_idx, (ev, tp, num, lbl, L, nums_full) in enumerate(test_loader):
        probs_seq = model.forward_sequence(ev, tp, num, L)
        for i in range(probs_seq.size(0)):
            probs = probs_seq[i][:L[i]]
            events_seq = ev[i][:L[i]]
            types_seq = tp[i][:L[i]]
            numerics_seq = nums_full[i][:L[i]]

            for t in range(1, len(probs)):
                delta_ev = expected_value(probs[t].cpu().numpy()) - expected_value(probs[t-1].cpu().numpy())
                event_name = inv_event_vocab.get(events_seq[t].item(), "UNKNOWN")
                type_name = inv_type_vocab.get(types_seq[t].item(), "UNKNOWN")
                x, y = numerics_seq[t][1].item(), numerics_seq[t][2].item()
                team_role = numerics_seq[t][0].item()

                # Removed outcome tag
                event_key = f"{event_name}"
                key_et = f"{event_name} | {type_name}"
                key_et_loc = f"{event_name} | {type_name} | ({x:.3f}, {y:.3f})"

                if team_role == 1:
                    event_oteam_ev_deltas[event_key].append(delta_ev)
                    event_type_oteam_ev_deltas[key_et].append(delta_ev)
                    event_type_loc_oteam_deltas[key_et_loc].append(delta_ev)
                elif team_role == -1:
                    event_dteam_ev_deltas[event_key].append(delta_ev)
                    event_type_dteam_ev_deltas[key_et].append(delta_ev)
                    event_type_loc_dteam_ev_deltas[key_et_loc].append(delta_ev)

def top_avg(d, name, role, apply_min=True):
    return pd.DataFrame([
        {"category": f"Top 10 {name} ({role})", "key": k, "avg_delta_ev": np.mean(v), "samples": len(v)}
        for k, v in d.items() if (len(v) >= MIN_SAMPLES if apply_min else True)
    ]).sort_values("avg_delta_ev", ascending=False).head(10)

def bottom_avg(d, name, role, apply_min=True):
    return pd.DataFrame([
        {"category": f"Bottom 10 {name} ({role})", "key": k, "avg_delta_ev": np.mean(v), "samples": len(v)}
        for k, v in d.items() if (len(v) >= MIN_SAMPLES if apply_min else True)
    ]).sort_values("avg_delta_ev", ascending=True).head(10)

final_df = pd.concat([
    top_avg(event_oteam_ev_deltas, "event", "O"),
    bottom_avg(event_oteam_ev_deltas, "event", "O"),
    top_avg(event_dteam_ev_deltas, "event", "D"),
    bottom_avg(event_dteam_ev_deltas, "event", "D"),

    top_avg(event_type_oteam_ev_deltas, "event_type", "O"),
    bottom_avg(event_type_oteam_ev_deltas, "event_type", "O"),
    top_avg(event_type_dteam_ev_deltas, "event_type", "D"),
    bottom_avg(event_type_dteam_ev_deltas, "event_type", "D"),

    top_avg(event_type_loc_oteam_deltas, "event_type_location", "O", apply_min=False),
    bottom_avg(event_type_loc_oteam_deltas, "event_type_location", "O", apply_min=False),
    top_avg(event_type_loc_dteam_ev_deltas, "event_type_location", "D", apply_min=False),
    bottom_avg(event_type_loc_dteam_ev_deltas, "event_type_location", "D", apply_min=False),
], ignore_index=True)

final_df.to_csv("Top_EV_Impact_Combos.csv", index=False)
print("✅ Saved: Top_EV_Impact_Combos.csv (no outcome tags)")






