# ==========================================================
# train_model.py – Full Training Script Using Shared Data Loader
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR

from model_def import HockeyModel, HockeyDS, collate
from data_loader import (
    X_train, X_dev, X_test,
    y_train, y_dev, y_test,
    event_vocab, type_vocab
)

# ==========================================================
# DataLoaders
# ==========================================================
train_loader = DataLoader(HockeyDS(X_train, y_train), batch_size=32, shuffle=True, collate_fn=collate)
dev_loader   = DataLoader(HockeyDS(X_dev, y_dev), batch_size=32, shuffle=False, collate_fn=collate)

# ==========================================================
# Model, Loss, Optimizer, Scheduler
# ==========================================================
model = HockeyModel(event_vocab_size=len(event_vocab)+1, type_vocab_size=len(type_vocab)+1)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = OneCycleLR(opt, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=30, pct_start=0.3)

# ==========================================================
# Training Loop
# ==========================================================
if __name__ == "__main__":
    for ep in range(1, 31):
        model.train()
        tot_loss = tot_corr = tot = 0
        for ev, tp, num, lbl, L, _ in train_loader:
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
        print(f"[Train] Epoch {ep:02d}/30 | Loss {tot_loss/tot:.4f} | Acc {tot_corr/tot:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "zone_exit_lstm_state_dict.pt")
    print("\nModel saved to 'zone_exit_lstm_state_dict.pt'")

