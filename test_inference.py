"""Quick test: load model and check predictions on a few examples."""

import argparse
import random

import torch
from modular_addition.tokenizer import ModularAdditionTokenizer


def load_model(path, device="cpu"):
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model


def build_inputs(tokenizer, operands_list, device="cpu"):
    return torch.tensor([
        [tokenizer.bos_token_id] + list(ops) + [tokenizer.eq_token_id]
        for ops in operands_list
    ], device=device)


def evaluate_outputs(tokenizer, operands_list, input_ids, outputs):
    correct = []
    wrong = []
    for ops, out in zip(operands_list, outputs):
        expected = sum(ops) % tokenizer.p
        generated = out[input_ids.shape[1]:].tolist()
        predicted = generated[0] if generated else None
        result = {"ops": ops, "expected": expected, "predicted": predicted, "decoded": tokenizer.decode(out.tolist(), skip_special_tokens=True)}
        if predicted == expected:
            correct.append(result)
        else:
            wrong.append(result)

    print(f"Correct: {len(correct)}/{len(operands_list)}")
    for r in correct:
        print(f"  ✓ {r['decoded']}")
    print(f"Wrong: {len(wrong)}/{len(operands_list)}")
    for r in wrong:
        print(f"  ✗ {r['decoded']} (expected {r['expected']}, got {r['predicted']})")

    return correct, wrong


def test_greedy(model, tokenizer, operands_list, device="cpu"):
    """Greedy generation (argmax at each step)."""
    input_ids = build_inputs(tokenizer, operands_list, device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return evaluate_outputs(tokenizer, operands_list, input_ids, outputs)


def test_sample(model, tokenizer, operands_list, device="cpu", temperature=1.0):
    """Sample from the distribution at T=temperature."""
    input_ids = build_inputs(tokenizer, operands_list, device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=True,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return evaluate_outputs(tokenizer, operands_list, input_ids, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    tokenizer = ModularAdditionTokenizer(args.p)
    model = load_model(args.model_path)

    pairs = [tuple(random.randint(0, tokenizer.p - 1) for _ in range(2)) for _ in range(args.n)]
    triples = [tuple(random.randint(0, tokenizer.p - 1) for _ in range(3)) for _ in range(args.n)]

    print("=== Greedy (2 operands) ===")
    test_greedy(model, tokenizer, pairs)

    print("\n=== Greedy (3 operands) ===")
    test_greedy(model, tokenizer, triples)

    print("\n=== T=1 Sampling (2 operands) ===")
    test_sample(model, tokenizer, pairs)

    print("\n=== T=1 Sampling (3 operands) ===")
    test_sample(model, tokenizer, triples)
