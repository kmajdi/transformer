import torch.nn as nn
from torch import Tensor, softmax
import torch
import math

class SDPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        size = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        q = torch.reshape(q, (size, q_len, self.d_hid))
        k = torch.reshape(k, (size, k_len, self.d_hid))
        v = torch.reshape(v, (size, v_len, self.d_hid))

        energy = torch.einsum("nhql,nlhd->nqhd", [q, k])

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = softmax(energy / (self.n_embed ** 0.5), dim=3)
        out = torch.reshape(torch.einsum("nhql,nlhd->nqhd", [attention, v]), (size, q_len, self.heads * self.d_hid))

        out = self.softmax(out)
        v = out @ v

        return v, out


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model

        self.lin_q = nn.Linear(self.d_model, self.d_model)
        self.lin_k = nn.Linear(self.d_model, self.d_model)
        self.lin_v = nn.Linear(self.d_model, self.d_model)

        self.attention = SDPAttention()
        self.cct = nn.Linear(d_model, d_model)

    def concat(self, tensor: Tensor):
        b_size, heads, length, d_tensor = tensor.size()
        d_model = heads * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(b_size, length, d_model)
        return tensor

    def split(self, tensor):
        b_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_heads

        tensor = tensor.view(b_size, length, self.n_heads, d_tensor).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask:Tensor = None):
        q, k, v = self.lin_q(), self.lin_k(), self.lin_v()
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.cct(out)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x += self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalFeedForward(nn.Module):
    def __init__(self, d_model:int, d_hid:int, dropout:float):
        super().__init__()
        self.lin_1 = nn.Linear(d_model, d_hid)
        self.lin_2 = nn.Linear(d_hid, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_hid:int):
        super().__init__()

        self.attention = MultiheadAttention(d_model, n_heads)
        self.add_norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.feed_forward = PositionalFeedForward(d_model, d_hid, dropout)
        self.add_norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x:Tensor, source_max:Tensor):
        temp_x = x
        x = self.attention(q=x, k=x, v=x, mask=source_max)
        x = self.dropout_1(x)
        x = self.add_norm_1(x + temp_x)

        temp_x = x
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = self.add_norm_2(x + temp_x)

        return x

class EncoderStack(nn.Module):
    def __init__(self, d_model:int, n_tokens:int, n_layers:int, n_heads:int, dropout:float, d_hid:int):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, d_hid=d_hid) for h in range(n_layers)])


    def forward(self, x:Tensor, source_mask:Tensor):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, source_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_hid:int, dropout:float):
        super().__init__()
        self.masked_attention = MultiheadAttention(d_model, n_heads)
        self.attention = MultiheadAttention(d_model, n_heads)
        self.add_norm_1 = nn.LayerNorm(d_model)
        self.add_norm_2 = nn.LayerNorm(d_model)
        self.add_norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.feed_forward = PositionalFeedForward(d_model, d_hid, dropout)

    def forward(self, dec:Tensor, enc:Tensor, target_mask:Tensor, source_mask:Tensor):
        temp_x = dec
        x = self.masked_attention(q=dec, k=dec, v=dec, mask=target_mask)
        x = self.dropout_1(x)
        x = self.add_norm_1(x + temp_x)

        if enc is not None:
            temp_x = x
            x = self.attention(q=x, k=enc, v=enc, mask=source_mask)
            x = self.dropout_2(x)
            x = self.add_norm_2(x + temp_x)

        temp_x = x
        x = self.feed_forward(x)
        x = self.dropout_3(x)
        x = self.add_norm_2(x + temp_x)

        return x

class DecoderStack(nn.Module):
    def __init__(self, d_model:int, n_tokens:int, n_layers:int, n_heads:int, dropout:float, d_hid:int):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, d_hid=d_hid) for h in range(n_layers)])
        self.lin = nn.Linear(d_model, n_tokens)

    def forward(self, target, source, target_mask, source_mask):
        target = self.emb(target)
        for layer in self.layers:
            target = layer(target, source, target_mask, source_mask)

        out = self.lin(target)
        return out

class Transformer(nn.Module):
    def __init__(self,
                 n_token:int,
                 n_heads:int,
                 d_model:int = 512,
                 max_length:int = 5000,
                 dropout:float = 0.1,
                 enc_layers:int = 8,
                 dec_layers:int = 8,
                 d_hid:int = 8
                 ):
        super().__init__()
        self.embedding = nn.Embedding(n_token, d_model)
        self.pos = PositionalEncoding(d_model, max_length, dropout)
        self.encoder = EncoderStack(d_model, n_token, enc_layers, n_heads, dropout, d_hid)
        self.decoder = DecoderStack(d_model, n_token, dec_layers, n_heads, dropout, d_hid)
        self.lin = nn.Linear(d_model, n_token)
    def make_target_mask(self, target:Tensor):
        b_size, target_len = target.size()
        target_mask = torch.tril(torch.ones((target_len, target_len)).expand(
            b_size, 1, target_len, target_len
        ))
        return target_mask

    def decode(self, source:Tensor, target:Tensor):
        target_mask = self.make_target_masks(target)
        enc_out = self.encoder(source)
        out_labels = []
        out = target
        b_size, length = source.size(0), source.size(1)

        for i in range(length):
            out = self.decoder(out, enc_out, target_mask)
            out = out[:, -1, :]
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, 0)

        return out_labels

    def forward(self, source:Tensor, target:Tensor, source_mask:Tensor=None):
        target_mask = self.make_target_mask(target)
        source = self.embedding(source)
        source = self.pos(source)
        enc_out = self.encoder(source)

        target = self.embedding(target)
        target = self.pos(target)

        out = self.decoder(target, enc_out, target_mask, source_mask)
        out = self.lin(out)
        return out

