# ==========================================================
# ev_analysis.py – Player Ratings Metric
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

# Inverse vocab lookup
inv_event_vocab = {v: k for k, v in event_vocab.items()}
inv_type_vocab  = {v: k for k, v in type_vocab.items()}

#  Custom EV weights from prior work
label_to_ev = {
    0: 0.38,   # fail
    1: 0.16,   # uncontrolled
    2: 0.06    # controlled
}

# Match those same weights when converting model probabilities to EV
def expected_value(probs):
    controlled, uncontrolled, fail = probs
    return (controlled * 0.38) + (uncontrolled * 0.16) + (fail * 0.06)

MIN_SAMPLES = 30

# Containers
player_oteam_ev_deltas = defaultdict(list)
player_dteam_ev_deltas = defaultdict(list)

# Reload test loader
test_loader = DataLoader(HockeyDS(X_test, y_test), batch_size=32, shuffle=False, collate_fn=collate)

# Main loop to track EV deltas
with torch.no_grad():
    for batch_idx, (ev, tp, num, lbl, L, nums_full) in enumerate(test_loader):
        probs_seq = model.forward_sequence(ev, tp, num, L)
        for i in range(probs_seq.size(0)):
            probs = probs_seq[i][:L[i]]
            players_seq = players_test[batch_idx * test_loader.batch_size + i]
            events_seq = ev[i][:L[i]]
            types_seq = tp[i][:L[i]]
            numerics_seq = nums_full[i][:L[i]]

            for t in range(1, len(probs)):
                if t == len(probs) - 1:
                    # Final delta uses ground truth vs model's *actual* final EV
                    true_label = lbl[i].item()
                    true_ev = label_to_ev.get(int(true_label), 0.0)
                    pred_ev = expected_value(probs[t].cpu().numpy())  
                    delta_ev = true_ev - pred_ev
                else:
                    delta_ev = expected_value(probs[t].cpu().numpy()) - expected_value(probs[t - 1].cpu().numpy())

                player = players_seq[t]
                team_role = numerics_seq[t][0].item()

                if team_role == 1:
                    player_oteam_ev_deltas[player].append(delta_ev)
                elif team_role == -1:
                    player_dteam_ev_deltas[player].append(delta_ev)

# Save raw player D-team EV scores
player_o_scores = pd.DataFrame([
    {"player_id": pid, "avg_delta_ev": np.mean(deltas), "samples": len(deltas)}
    for pid, deltas in player_oteam_ev_deltas.items()
    if len(deltas) >= MIN_SAMPLES
])
player_d_scores = pd.DataFrame([
    {"player_id": pid, "avg_delta_ev": np.mean(deltas), "samples": len(deltas)}
    for pid, deltas in player_dteam_ev_deltas.items()
    if len(deltas) >= MIN_SAMPLES
])


player_d_scores.to_csv("Player_D_Team_EV_Scores.csv", index=False)



