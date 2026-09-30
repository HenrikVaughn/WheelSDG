from __future__ import annotations

import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

DATA_FILE = "input.txt"

BLOCK_SIZE = 64
BATCH_SIZE = 8
EVAL_BATCHES = 20
VOCAB_SIZE_EXPECTED = 65


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_head,
        block_size,
    ):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                "d_model must be divisible by n_head."
            )

        self.n_head = n_head
        self.d_model = d_model

        self.c_attn = nn.Linear(
            d_model,
            3 * d_model,
        )

        self.c_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size,
                )
            ).view(
                1,
                1,
                block_size,
                block_size,
            ),
        )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(
            self.d_model,
            dim=2,
        )

        head_dim = C // self.n_head

        k = k.view(
            B,
            T,
            self.n_head,
            head_dim,
        ).transpose(1, 2)

        q = q.view(
            B,
            T,
            self.n_head,
            head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            head_dim,
        ).transpose(1, 2)

        att = (
            q @ k.transpose(-2, -1)
        ) * (
            1.0 / math.sqrt(head_dim)
        )

        att = att.masked_fill(
            self.bias[
                :,
                :,
                :T,
                :T,
            ] == 0,
            float("-inf"),
        )

        att = F.softmax(
            att,
            dim=-1,
        )

        y = att @ v

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.c_fc = nn.Linear(
            d_model,
            4 * d_model,
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * d_model,
            d_model,
        )

    def forward(self, x):

        return self.c_proj(
            self.gelu(
                self.c_fc(x)
            )
        )


class Block(nn.Module):

    def __init__(
        self,
        d_model,
        n_head,
        block_size,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            d_model
        )

        self.attn = CausalSelfAttention(
            d_model,
            n_head,
            block_size,
        )

        self.ln_2 = nn.LayerNorm(
            d_model
        )

        self.mlp = MLP(
            d_model
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size=65,
        block_size=64,
        d_model=144,
        n_layer=4,
        n_head=4,
    ):
        super().__init__()

        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    vocab_size,
                    d_model,
                ),
                wpe=nn.Embedding(
                    block_size,
                    d_model,
                ),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model,
                            n_head,
                            block_size,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(
                    d_model
                ),
            )
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        idx,
        targets=None,
    ):

        device = idx.device

        B, T = idx.size()

        if T > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length "
                f"{T}, block size is {self.block_size}"
            )

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=device,
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1),
                ),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.85,
        top_k=40,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[
                :,
                -self.block_size:
            ]

            logits, _ = self(idx_cond)

            logits = logits[
                :,
                -1,
                :
            ]

            temperature = max(
                temperature,
                1e-5,
            )

            logits = logits / temperature

            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                minimum = values[:, -1:]

                logits = torch.where(
                    logits < minimum,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            idx = torch.cat(
                [
                    idx,
                    next_token,
                ],
                dim=1,
            )

        return idx


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def download_dataset():

    if os.path.exists(DATA_FILE):
        return

    print("Downloading TinyShakespeare...")

    urllib.request.urlretrieve(
        DATA_URL,
        DATA_FILE,
    )

    print("Download complete.")


def load_dataset():

    download_dataset()

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        text = file.read()

    characters = sorted(set(text))

    stoi = {
        c: i
        for i, c in enumerate(characters)
    }

    itos = {
        i: c
        for i, c in enumerate(characters)
    }

    encoded = torch.tensor(
        [
            stoi[c]
            for c in text
        ],
        dtype=torch.long,
    )

    split = int(
        0.90 * len(encoded)
    )

    train_data = encoded[:split]
    val_data = encoded[split:]

    return (
        text,
        train_data,
        val_data,
        stoi,
        itos,
        len(characters),
    )


def get_batch(
    data,
    device,
):

    max_start = (
        len(data)
        - BLOCK_SIZE
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack(
        [
            data[
                start:
                start + BLOCK_SIZE
            ]
            for start in starts.tolist()
        ]
    )

    y = torch.stack(
        [
            data[
                start + 1:
                start + BLOCK_SIZE + 1
            ]
            for start in starts.tolist()
        ]
    )

    return (
        x.to(device),
        y.to(device),
    )


def create_eval_batches(
    data,
    seed,
    device,
):

    generator = torch.Generator()

    generator.manual_seed(seed)

    batches = []

    max_start = (
        len(data)
        - BLOCK_SIZE
        - 1
    )

    for _ in range(EVAL_BATCHES):

        starts = torch.randint(
            0,
            max_start,
            (BATCH_SIZE,),
            generator=generator,
        )

        x = torch.stack(
            [
                data[
                    start:
                    start + BLOCK_SIZE
                ]
                for start in starts.tolist()
            ]
        )

        y = torch.stack(
            [
                data[
                    start + 1:
                    start + BLOCK_SIZE + 1
                ]
                for start in starts.tolist()
            ]
        )

        batches.append(
            (
                x.to(device),
                y.to(device),
            )
        )

    return batches


@torch.no_grad()
def evaluate_fixed(
    model,
    batches,
):

    model.eval()

    losses = []

    for x, y in batches:

        _, loss = model(
            x,
            y,
        )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)