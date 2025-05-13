# ==========================================================
# model_def.py – Model and Dataset Definitions Only
# ==========================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# ---------- WeightDrop wrapper ----------
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

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)

# ---------- Model Definition ----------
class HockeyModel(nn.Module):
    def __init__(self, event_vocab_size=1, type_vocab_size=1):
        super().__init__()
        emb_dim = {"event": 16, "type": 8}
        numeric_dim = 6
        hidden_size = 64
        num_layers = 2
        dropout = 0.3

        self.event_emb = nn.Embedding(event_vocab_size, emb_dim["event"], padding_idx=0)
        self.type_emb  = nn.Embedding(type_vocab_size,  emb_dim["type"],  padding_idx=0)
        input_dim = sum(emb_dim.values()) + numeric_dim
        base_lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.lstm = WeightDrop(base_lstm, weights=('weight_hh_l0',), dropout=dropout)
        self.attn_lin = nn.Linear(hidden_size, hidden_size)
        self.attn_vec = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 3)

    def forward(self, ev, tp, num, lengths):
        x = torch.cat([self.event_emb(ev), self.type_emb(tp), num], dim=-1)
        packed_out, _ = self.lstm(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        u = torch.tanh(self.attn_lin(out))
        scores = torch.matmul(u, self.attn_vec)
        T = out.size(1)
        mask = (torch.arange(T, device=lengths.device)[None, :] < lengths[:, None])
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(w * out, dim=1)
        z = self.dropout(self.layernorm(context))
        return self.fc(z)

    def forward_sequence(self, ev, tp, num, lengths):
        x = torch.cat([self.event_emb(ev), self.type_emb(tp), num], dim=-1)
        packed_out, _ = self.lstm(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        z = self.dropout(self.layernorm(out))
        return torch.softmax(self.fc(z), dim=-1)

# ---------- Dataset Definition ----------
class HockeyDS(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = torch.tensor(self.seqs[idx], dtype=torch.float)
        L = s.size(0)
        return (
            s[:, 5].long(),          # event_idx
            s[:, 6].long(),          # type_idx
            s[:, :6],                # numeric features
            torch.tensor(self.labels[idx], dtype=torch.long),
            L,
            s                       
        )


# ---------- Collate Function ----------
def collate(batch):
    ev, tp, num, lbl, Ls, nums_full = zip(*batch)
    ev  = pad_sequence(ev,  batch_first=True, padding_value=0)
    tp  = pad_sequence(tp,  batch_first=True, padding_value=0)
    num = pad_sequence(num, batch_first=True, padding_value=0.0)
    nums_full = pad_sequence(nums_full, batch_first=True, padding_value=0.0)
    return ev, tp, num, torch.stack(lbl), torch.tensor(Ls), nums_full
