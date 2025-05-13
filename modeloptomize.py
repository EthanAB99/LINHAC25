"""
This code implements a Long Short-Term Memory (LSTM) neural network to predict hockey zone 
exit outcomes (successful controlled, successful uncontrolled, or failed) using hockey game 
event-sequence data with features including talent rating, team possession, x/y coordinates, 
player position, and scaled score differential. It performs a grid search over hyperparameters 
(e.g., embedding sizes, hidden size, learning rate) to identify optimal settings, using a 
second-order regression model to refine parameter selection. The LSTM, enhanced with WeightDrop 
regularization, label smoothing loss, and an attention mechanism, is trained with the Adam 
optimizer and OneCycleLR scheduler over 25 epochs, evaluating performance on training, 
validation, and test sets. Final test evaluation reports loss, accuracy, a confusion matrix, 
and detailed classification metrics.



Optimal Params: {'event_emb': np.float64(46.079550969092566), 'type_emb': 
np.float64(18.720384790911325), 'hidden_size': np.float64(151.03942360363115), 
'lr': np.float64(0.0008550455423619569), 'weight_decay': np.float64(1e-06), 
'batch_size': np.float64(41.921410986262195), 'dropout': np.float64(0.0), 
'num_layers': np.float64(2.080436548530037), 'smoothing': np.float64(0.3)}
"""
# 
#  ==========================================================
# Zone-Exit LSTM – FINAL WORKING VERSION (with Grid Search, Fixed)
# ==========================================================

# ---------- imports ----------
import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import itertools
import random

# ---------- Label Smoothing Loss ----------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.dim = dim

    def forward(self, pred, target):
        log_probs = nn.functional.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (pred.size(self.dim) - 1))
            true_dist.scatter_(self.dim, target.unsqueeze(self.dim), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=self.dim))

# ---------- WeightDrop ----------
class WeightDrop(nn.Module):
    def __init__(self, module, weights=('weight_hh_l0',), dropout=0.25):
        super().__init__()
        self.module, self.weights, self.dropout = module, weights, dropout
        for w in weights:
            raw = getattr(self.module, w)
            self.register_parameter(w + "_raw", nn.Parameter(raw.data))
            del self.module._parameters[w]
    def _setweights(self):
        for w in self.weights:
            raw = getattr(self, w + "_raw")
            dropped = nn.functional.dropout(raw, p=self.dropout, training=self.training)
            setattr(self.module, w, nn.Parameter(dropped))
    def forward(self, *args, **kw):
        self._setweights()
        return self.module(*args, **kw)

# ---------- 1) load & preprocess ----------
data = pd.read_csv("training2.csv")
data["events"] = data["events"].apply(json.loads)
data["result"] = data["result"].map({"SC": 1, "SD": 2, "F": 0})

# ---------- 2) build vocabs ----------
event_vocab = {ev: i+1 for i, ev in enumerate(
    sorted({e["eventname"] for s in data["events"] for e in s})
)}
type_vocab = {tp: i+1 for i, tp in enumerate(
    sorted({e["type"]      for s in data["events"] for e in s})
)}
event_vocab_size = len(event_vocab) + 1
type_vocab_size  = len(type_vocab)  + 1

# ---------- 3) map indices ----------
for seq in data["events"]:
    for e in seq:
        e["event_idx"] = event_vocab.get(e["eventname"], 0)
        e["type_idx"]  = type_vocab.get(e["type"], 0)

# ---------- 4) feature extraction ----------
max_seq_len = 30
def extract_features(seq):
    feats = []
    for e in seq:
        feats.append([
            float(e.get("talent_rating", 0)),
            float(e.get("teaminpossession", 0)),
            float(e.get("x", 0)),
            float(e.get("y", 0)),
            int(e.get("event_idx", 0)),
            int(e.get("type_idx", 0)),
            int(e.get("playerprimaryposition", 0))
                 if str(e.get("playerprimaryposition", 0)).isdigit() else 0,
            float(e.get("scorediff_scaled", 0)),
        ])
    return feats

X, y = [], []
for seq, lbl in zip(data["events"], data["result"]):
    feats = extract_features(seq)
    if len(feats) <= max_seq_len:
        X.append(feats)
        y.append(lbl)

# ---------- 5) train/dev/test split ----------
X_trval, X_test, y_trval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_trval, y_trval, test_size=0.1765, stratify=y_trval, random_state=42
)

# ---------- 6) Dataset & DataLoader ----------
class HockeyDS(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = torch.tensor(self.seqs[idx], dtype=torch.float)
        L = s.size(0)
        return (
            s[:,4].long(),           # event idx
            s[:,5].long(),           # type  idx
            s[:,[0,1,2,3,6,7]],      # six numeric features
            torch.tensor(self.labels[idx], dtype=torch.long),
            L
        )

def collate(batch):
    ev,tp,num,lbl,Ls = zip(*batch)
    ev  = pad_sequence(ev,  batch_first=True, padding_value=0)
    tp  = pad_sequence(tp,  batch_first=True, padding_value=0)
    num = pad_sequence(num, batch_first=True, padding_value=0.0)
    return ev, tp, num, torch.stack(lbl), torch.tensor(Ls)

# ---------- 7) model definition ----------
class HockeyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_emb = nn.Embedding(event_vocab_size, 16, padding_idx=0)
        self.type_emb  = nn.Embedding(type_vocab_size,  8,  padding_idx=0)
        input_dim = 16 + 8 + 6
        base_lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.3)
        self.lstm = WeightDrop(base_lstm, weights=('weight_hh_l0',), dropout=0.25)
        self.attn_lin = nn.Linear(64, 64)
        self.attn_vec = nn.Parameter(torch.empty(64))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        self.layernorm = nn.LayerNorm(64)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(64, 3)

    def forward(self, ev, tp, num, lengths):
        x = torch.cat([self.event_emb(ev), self.type_emb(tp), num], dim=-1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        u = torch.tanh(self.attn_lin(out))
        scores = torch.matmul(u, self.attn_vec)
        T = out.size(1)
        mask = (torch.arange(T, device=lengths.device)[None,:] < lengths[:,None])
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(w * out, dim=1)
        z = self.dropout(self.layernorm(context))
        return self.fc(z)

# ---------- 8) Grid Search ----------
param_grid = {
    'event_emb': [16, 32, 64],
    'type_emb': [8, 16, 32],
    'hidden_size': [64, 128, 256],
    'lr': [1e-4, 5e-4, 1e-3],
    'weight_decay': [1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'dropout': [0.0, 0.2, 0.4],
    'num_layers': [1, 2, 3],
    'smoothing': [0.0, 0.1, 0.2]
}
combinations = list(itertools.product(*param_grid.values()))
random.shuffle(combinations)
results = []
numeric_dim = 6

for i, (event_emb, type_emb, hidden_size, lr, weight_decay, batch_size, dropout, num_layers, smoothing) in enumerate(combinations[:50], 1):
    print(f"Trial {i}/50: event_emb={event_emb}, type_emb={type_emb}, hidden_size={hidden_size}, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}, dropout={dropout}, num_layers={num_layers}, smoothing={smoothing}")
    emb_dims = {"event": event_emb, "type": type_emb}
    train_loader = DataLoader(
        HockeyDS(X_train, y_train),
        batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        HockeyDS(X_dev, y_dev),
        batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    model = HockeyModel()
    model.event_emb = nn.Embedding(event_vocab_size, event_emb, padding_idx=0)
    model.type_emb = nn.Embedding(type_vocab_size, type_emb, padding_idx=0)
    input_dim = event_emb + type_emb + numeric_dim
    base_lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                        batch_first=True, dropout=dropout if num_layers > 1 else 0)
    model.lstm = WeightDrop(base_lstm, weights=('weight_hh_l0',), dropout=dropout)
    model.attn_lin = nn.Linear(hidden_size, hidden_size)
    model.attn_vec = nn.Parameter(torch.empty(hidden_size))
    nn.init.xavier_uniform_(model.attn_vec.unsqueeze(0))
    model.layernorm = nn.LayerNorm(hidden_size)
    model.dropout = nn.Dropout(dropout)
    model.fc = nn.Linear(hidden_size, 3)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = LabelSmoothingLoss(smoothing=smoothing)
    best_dev_acc = 0
    for epoch in range(10):
        model.train()
        for ev, tp, num, lbl, L in train_loader:
            opt.zero_grad()
            out = model(ev, tp, num, L)
            loss = criterion(out, lbl)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        dev_correct = dev_total = 0
        with torch.no_grad():
            for ev, tp, num, lbl, L in dev_loader:
                out = model(ev, tp, num, L)
                preds = out.argmax(1)
                dev_correct += (preds == lbl).sum().item()
                dev_total += lbl.size(0)
        dev_acc = dev_correct / dev_total
        best_dev_acc = max(best_dev_acc, dev_acc)
    results.append([event_emb, type_emb, hidden_size, lr, weight_decay, batch_size, dropout, num_layers, smoothing, best_dev_acc])
    print(f"Dev Acc: {best_dev_acc:.4f}")

# Save Grid Search Results
results_df = pd.DataFrame(
    results,
    columns=['event_emb', 'type_emb', 'hidden_size', 'lr', 'weight_decay', 'batch_size', 'dropout', 'num_layers', 'smoothing', 'dev_acc']
)
results_df.to_csv("grid_search_results.csv", index=False)
print("Grid search results saved to grid_search_results.csv")

# Second-Order Regression
results = np.array(results)
X = results[:, :-1]
y = results[:, -1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)
reg = LinearRegression().fit(X_poly, y)

def predict_acc(params):
    X_scaled = scaler.transform([params])
    X_poly = poly.transform(X_scaled)
    return reg.predict(X_poly)[0]

def objective(params):
    return -predict_acc(params)

bounds = [(8, 64), (4, 32), (32, 256), (1e-5, 1e-2), (1e-6, 1e-3), (8, 64), (0.0, 0.5), (1, 3), (0.0, 0.3)]
result = minimize(objective, np.mean(X, axis=0), bounds=bounds, method='L-BFGS-B')
optimal_params = result.x
print("Optimal Params:", dict(zip(param_grid.keys(), optimal_params)))

# ---------- 9) model with optimal parameters ----------
emb_dims = {"event": int(optimal_params[0]), "type": int(optimal_params[1])}
hidden_size = int(optimal_params[2])
batch_size = int(optimal_params[5])
dropout = optimal_params[6]
num_layers = int(optimal_params[7])
smoothing = optimal_params[8]

train_loader = DataLoader(HockeyDS(X_train, y_train), batch_size, True, collate_fn=collate)
dev_loader = DataLoader(HockeyDS(X_dev, y_dev), batch_size, False, collate_fn=collate)
test_loader = DataLoader(HockeyDS(X_test, y_test), batch_size, False, collate_fn=collate)

model = HockeyModel()
model.event_emb = nn.Embedding(event_vocab_size, emb_dims["event"], padding_idx=0)
model.type_emb = nn.Embedding(type_vocab_size, emb_dims["type"], padding_idx=0)
input_dim = sum(emb_dims.values()) + numeric_dim
base_lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
model.lstm = WeightDrop(base_lstm, weights=('weight_hh_l0',), dropout=dropout)
model.attn_lin = nn.Linear(hidden_size, hidden_size)
model.attn_vec = nn.Parameter(torch.empty(hidden_size))
nn.init.xavier_uniform_(model.attn_vec.unsqueeze(0))
model.layernorm = nn.LayerNorm(hidden_size)
model.dropout = nn.Dropout(dropout)
model.fc = nn.Linear(hidden_size, 3)

# ---------- 10) loss, optimizer, scheduler ----------
criterion = LabelSmoothingLoss(smoothing=smoothing)
opt = optim.Adam(model.parameters(), lr=optimal_params[3], weight_decay=optimal_params[4])
num_epochs = 25
sched = OneCycleLR(opt,
                   max_lr=optimal_params[3]*2,
                   steps_per_epoch=len(train_loader),
                   epochs=num_epochs,
                   pct_start=0.3)

# ---------- 11) training ----------
for ep in range(1, num_epochs+1):
    model.train()
    tot_loss = tot_corr = tot = 0
    for ev, tp, num, lbl, L in train_loader:
        opt.zero_grad()
        out = model(ev, tp, num, L)
        loss = criterion(out, lbl)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        bs = lbl.size(0)
        tot += bs
        tot_loss += loss.item() * bs
        tot_corr += (out.argmax(1) == lbl).sum().item()
    print(f"Epoch {ep}/{num_epochs} | Train Loss {tot_loss/tot:.4f} | Train Acc {tot_corr/tot:.4f}")

    model.eval()
    d_loss = d_corr = d_tot = 0
    with torch.no_grad():
        for ev, tp, num, lbl, L in dev_loader:
            out = model(ev, tp, num, L)
            ls  = criterion(out, lbl)
            b   = lbl.size(0)
            d_tot += b
            d_loss += ls.item() * b
            d_corr += (out.argmax(1) == lbl).sum().item()
    print(f"           Dev Loss {d_loss/d_tot:.4f} | Dev Acc {d_corr/d_tot:.4f}")

# ---------- 12) test evaluation ----------
model.eval()
t_loss = t_corr = t_tot = 0
y_true = []
y_pred = []
with torch.no_grad():
    for ev, tp, num, lbl, L in test_loader:
        out = model(ev, tp, num, L)
        ls  = criterion(out, lbl)
        b   = lbl.size(0)
        t_tot += b
        t_loss += ls.item() * b
        t_corr += (out.argmax(1) == lbl).sum().item()
        y_true += lbl.cpu().tolist()
        y_pred += out.argmax(1).cpu().tolist()

print(f"\nTest Loss {t_loss/t_tot:.4f} | Test Acc {t_corr/t_tot:.4f}\n")
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))