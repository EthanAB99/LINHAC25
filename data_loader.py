# ==========================================================
# data_loader.py – Shared Preprocessing for Training + Evaluation
# ==========================================================

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Load & Preprocess ----------
data = pd.read_csv("training2.csv")
data["events"] = data["events"].apply(json.loads)
data["result"] = data["result"].map({"SC": 1, "SD": 2, "F": 0})

# ---------- Build Vocabularies ----------
event_vocab = {ev: i + 1 for i, ev in enumerate(
    sorted({e["eventname"] for s in data["events"] for e in s})
)}
type_vocab = {tp: i + 1 for i, tp in enumerate(
    sorted({e["type"] for s in data["events"] for e in s})
)}

# ---------- Map to Indices ----------
for seq in data["events"]:
    for e in seq:
        e["event_idx"] = event_vocab.get(e["eventname"], 0)
        e["type_idx"] = type_vocab.get(e["type"], 0)

# ---------- Feature Extraction ----------
max_seq_len = 30
def extract_features(seq):
    feats = []
    players = []
    for e in seq:
        feats.append([
            float(e.get("teamid", 0)),
            float(e.get("xadjcoord", 0)),
            float(e.get("yadjcoord", 0)),
            int(e.get("playerprimaryposition", 0)) if str(e.get("playerprimaryposition", 0)).isdigit() else 0,
            float(e.get("scorediff_scaled", 0)),
            int(e["event_idx"]),
            int(e["type_idx"]),
            float(e.get("outcome", 0.0))
        ])
        players.append(e.get("playerid", "UNKNOWN"))
    if len(feats) > max_seq_len:
        feats = feats[-max_seq_len:]
        players = players[-max_seq_len:]
    return feats, players

# ---------- Build X, y, players ----------
X = []
y = []
players_all = []
for seq, lbl in zip(data["events"], data["result"]):
    feats, players = extract_features(seq)
    if feats:
        X.append(feats)
        y.append(lbl)
        players_all.append(players)

# ---------- Train/Dev/Test Split ----------
X_trval, X_test, y_trval, y_test, players_trval, players_test = train_test_split(
    X, y, players_all, test_size=0.15, stratify=y, random_state=42
)
X_train, X_dev, y_train, y_dev, players_train, players_dev = train_test_split(
    X_trval, y_trval, players_trval, test_size=0.1765, stratify=y_trval, random_state=42
)

# ---------- Exportables ----------
__all__ = [
    "X_train", "X_dev", "X_test",
    "y_train", "y_dev", "y_test",
    "players_test", "event_vocab", "type_vocab"
]
