from __future__ import annotations

import os
import random
import time

import torch

from math_framework import (
    GPT,
    BLOCK_SIZE,
    BATCH_SIZE,
    VOCAB_SIZE_EXPECTED,
    count_parameters,
    create_eval_batches,
    evaluate_fixed,
    get_batch,
    load_dataset,
)

from optimizer import (
    AttentionMLPWheelOptimizer,
)


SEED = 1337

DEVICE = torch.device("cpu")

D_MODEL = 144
N_LAYER = 4
N_HEAD = 4

MAX_STEPS = 1500

BASE_INFINITESIMAL = 2e-3
PROJECTIVE_CEILING = 6.0
WEIGHT_DECAY = 1e-2

ALPHA = 0.19

LOG_INTERVAL_SECONDS = 30.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 20

GENERATE_TOKENS = 500
TEMPERATURE = 0.85
TOP_K = 40

CHECKPOINT_FILE = (
    "hyperwheel_tinyshakespeare_best.pt"
)


random.seed(SEED)

torch.manual_seed(SEED)

try:
    torch.set_num_threads(
        max(
            1,
            os.cpu_count() or 1,
        )
    )
except Exception:
    pass


def main():

    print("=" * 72)
    print("HyperWheel + TinyShakespeare")
    print("=" * 72)

    print(
        f"Device:              {DEVICE}"
    )

    print(
        f"Context:             {BLOCK_SIZE}"
    )

    print(
        f"Batch size:          {BATCH_SIZE}"
    )

    print(
        f"Model width:         {D_MODEL}"
    )

    print(
        f"Layers:              {N_LAYER}"
    )

    print(
        f"Heads:               {N_HEAD}"
    )

    print(
        f"Training steps:      {MAX_STEPS}"
    )

    print(
        f"Base infinitesimal:  "
        f"{BASE_INFINITESIMAL:.4e}"
    )

    print(
        f"Projective ceiling:  "
        f"{PROJECTIVE_CEILING:.4f}"
    )

    print(
        f"Weight decay:        "
        f"{WEIGHT_DECAY:.4e}"
    )

    print(
        f"Global/local alpha:  "
        f"{ALPHA:.4f}"
    )

    print("=" * 72)

    (
        text,
        train_data,
        val_data,
        stoi,
        itos,
        vocab_size,
    ) = load_dataset()

    print(
        f"Corpus chars:        "
        f"{len(text):,}"
    )

    print(
        f"Vocabulary:          "
        f"{vocab_size}"
    )

    if vocab_size != VOCAB_SIZE_EXPECTED:

        print("WARNING:")

        print(
            f"Expected roughly "
            f"{VOCAB_SIZE_EXPECTED} "
            f"characters, got "
            f"{vocab_size}."
        )

    print(
        f"Train chars:         "
        f"{len(train_data):,}"
    )

    print(
        f"Validation chars:    "
        f"{len(val_data):,}"
    )

    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    ).to(DEVICE)

    parameter_count = count_parameters(
        model
    )

    print(
        f"Parameters:          "
        f"{parameter_count:,}"
    )

    parameter_groups = []

    parameter_groups.append(
        {
            "params": [
                model.transformer.wte.weight,
                model.transformer.wpe.weight,
            ],
            "name": "embedding",
        }
    )

    for i, block in enumerate(
        model.transformer.h
    ):

        parameter_groups.append(
            {
                "params": (
                    list(
                        block.ln_1.parameters()
                    )
                    + list(
                        block.attn.parameters()
                    )
                ),
                "name": (
                    f"block_{i}_attention"
                ),
            }
        )

        parameter_groups.append(
            {
                "params": (
                    list(
                        block.ln_2.parameters()
                    )
                    + list(
                        block.mlp.parameters()
                    )
                ),
                "name": (
                    f"block_{i}_mlp"
                ),
            }
        )

    parameter_groups.append(
        {
            "params": (
                list(
                    model.transformer.ln_f.parameters()
                )
                + list(
                    model.lm_head.parameters()
                )
            ),
            "name": "final",
        }
    )

    optimizer = (
        AttentionMLPWheelOptimizer(
            parameter_groups,
            base_infinitesimal=(
                BASE_INFINITESIMAL
            ),
            projective_ceiling=(
                PROJECTIVE_CEILING
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            alpha=ALPHA,
        )
    )

    print(
        "Optimizer:            "
        "AttentionMLPWheelOptimizer"
    )

    print(
        "Optimizer state:      none"
    )

    print(
        "Momentum:             none"
    )

    print(
        "Second moment:        none"
    )

    print(
        "Gradient history:     none"
    )

    print("=" * 72)

    train_eval_batches = (
        create_eval_batches(
            train_data,
            seed=SEED + 1000,
            device=DEVICE,
        )
    )

    val_eval_batches = (
        create_eval_batches(
            val_data,
            seed=SEED + 2000,
            device=DEVICE,
        )
    )

    start_time = time.time()
    last_log_time = start_time

    best_val_loss = float("inf")

    model.train()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        x, y = get_batch(
            train_data,
            DEVICE,
        )

        optimizer.zero_grad()

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        optimizer.step()

        now = time.time()

        should_evaluate = (
            step == 1
            or step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        )

        should_log = (
            now - last_log_time
            >= LOG_INTERVAL_SECONDS
        )

        if (
            should_evaluate
            or should_log
        ):

            train_loss = evaluate_fixed(
                model,
                train_eval_batches,
            )

            val_loss = evaluate_fixed(
                model,
                val_eval_batches,
            )

            elapsed = (
                now - start_time
            )

            print(
                f"[{elapsed:8.1f}s] "
                f"Step {step:5d}/{MAX_STEPS} "
                f"| batch {loss.item():.4f} "
                f"| train {train_loss:.4f} "
                f"| val {val_loss:.4f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "step":
                            step,

                        "best_val_loss":
                            best_val_loss,

                        "vocab_size":
                            vocab_size,

                        "block_size":
                            BLOCK_SIZE,

                        "d_model":
                            D_MODEL,

                        "n_head":
                            N_HEAD,

                        "n_layer":
                            N_LAYER,

                        "stoi":
                            stoi,

                        "itos":
                            itos,

                        "optimizer_name":
                            "AttentionMLPWheelOptimizer",

                        "base_infinitesimal":
                            BASE_INFINITESIMAL,

                        "projective_ceiling":
                            PROJECTIVE_CEILING,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "alpha":
                            ALPHA,
                    },
                    CHECKPOINT_FILE,
                )

                print(
                    f"               "
                    f"| NEW BEST "
                    f"val={best_val_loss:.4f}"
                )

            last_log_time = now

    train_loss = evaluate_fixed(
        model,
        train_eval_batches,
    )

    val_loss = evaluate_fixed(
        model,
        val_eval_batches,
    )

    elapsed_total = (
        time.time() - start_time
    )

    print()

    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(
        f"Train loss:       "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation loss:  "
        f"{val_loss:.4f}"
    )

    print(
        f"Best validation:  "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Total time:       "
        f"{elapsed_total:.1f}s"
    )

    print(
        f"Avg time/step:    "
        f"{elapsed_total / MAX_STEPS:.4f}s"
    )

    print(
        f"Optimizer states: "
        f"{len(optimizer.state)}"
    )

    print(
        f"Alpha:             "
        f"{ALPHA:.4f}"
    )

    print(
        f"Checkpoint:        "
        f"{CHECKPOINT_FILE}"
    )

    print("=" * 72)

    model.eval()

    prompt = "ROMEO:"

    prompt_tensor = torch.tensor(
        [
            [
                stoi[c]
                for c in prompt
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    generated = model.generate(
        prompt_tensor,
        max_new_tokens=GENERATE_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    generated_text = "".join(
        itos[int(token)]
        for token in generated[0]
    )

    print()
    print("--- SAMPLE ---")
    print(generated_text)
    print("--- END SAMPLE ---")


if __name__ == "__main__":
    main()