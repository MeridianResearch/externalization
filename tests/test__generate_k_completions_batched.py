"""
Test suite for validating generate_k_completions_batched correctness.

This test suite compares the batched implementation against the sequential
implementation across multiple configurations to ensure identical outputs.
"""

import sys
from pathlib import Path
from copy import deepcopy

import pytest
import torch

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from early_exit.util import get_model, load_model_from_wandb, load_model
from early_exit.rl_utils import generate_k_completions, generate_k_completions_batched
from early_exit.patching import replace_attention_layers
from early_exit.rl_types import RLHyperparams
from shared_utils.load import get_tokenizer, configs_from_yaml


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def model_setup():
    """Setup model once for all tests."""
    print("\n" + "=" * 80)
    print("Loading model for tests (this may take a few minutes)...")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config_path = ROOT_DIR / "config_deepseek.yaml"
    sft_model_path = ROOT_DIR / "models/early_exit_20250908_layers_5_big"
    
    tokenizer = get_tokenizer(model_name)
    config = configs_from_yaml(config_path, tokenizer.eos_token_id)
    
    student = get_model(model_name, config['model'], device)
    student = replace_attention_layers(student, config['lora'], device)
    student = load_model_from_wandb(
        student,
        model_path=sft_model_path,
        artifact_path='vkarthik095-university-of-amsterdam/early-exit/early_exit_20250908_layers_5_big:v0'
    )
    student = load_model(student, sft_model_path)
    student.eval()
    
    print("✓ Model loaded successfully")
    print("=" * 80)
    
    return {
        'model': student,
        'tokenizer': tokenizer,
        'config': config,
        'device': device
    }


@pytest.fixture
def system_prompt():
    """Get system prompt from RLHyperparams."""
    return RLHyperparams().system_prompt


@pytest.fixture
def test_prompts():
    """Get standard test prompts."""
    return [
        "What is 15 multiplied by 23",
        "Solve for x: 2x + 5 = 17",
        "How many continents are there"
    ]


# ============================================================================
# Helper Functions
# ============================================================================

def print_generation_summary(completions, exit_info, label):
    """Print summary of generated completions."""
    print(f"\n{label} Generation Summary:")
    print(f"  - Tokens shape: {completions['tokens'].shape}")
    print(f"  - Number of texts: {len(completions['texts'])}")
    print(f"  - Number of exit layer sequences: {len(exit_info['prescribed_exit_layers'])}")
    if len(exit_info['prescribed_exit_layers']) > 0:
        first_exit = exit_info['prescribed_exit_layers'][0]
        print(f"  - First exit layer sequence length: {len(first_exit)}")
        print(f"  - First exit layer sequence (first 10): {first_exit[:10].tolist()}")


def assert_completions_equal(completions1, completions2, exit_info1, exit_info2, label1="sequential", label2="batched"):
    """Helper to assert two completion outputs are equal with detailed diagnostics."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED COMPARISON: {label1} vs {label2}")
    print(f"{'=' * 80}")
    
    # Check shapes
    print(f"\n1. Token Shape Comparison:")
    print(f"   {label1}: {completions1['tokens'].shape}")
    print(f"   {label2}: {completions2['tokens'].shape}")
    
    if completions1['tokens'].shape != completions2['tokens'].shape:
        print(f"   ❌ SHAPES DIFFER!")
        assert False, f"Token shapes differ: {completions1['tokens'].shape} vs {completions2['tokens'].shape}"
    else:
        print(f"   ✓ Shapes match")
    
    # Check token equality
    print(f"\n2. Token Content Comparison:")
    tokens_equal = torch.equal(completions1['tokens'], completions2['tokens'])
    print(f"   Tokens equal: {tokens_equal}")
    
    if not tokens_equal:
        # Find differences
        diff_mask = completions1['tokens'] != completions2['tokens']
        num_diffs = diff_mask.sum().item()
        print(f"   ❌ TOKENS DIFFER at {num_diffs} positions!")
        
        # Show first few differences
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print(f"\n   First 10 differences:")
        for idx in diff_indices[:10]:
            batch_idx, pos_idx = idx[0].item(), idx[1].item()
            print(f"     Position [{batch_idx}, {pos_idx}]: {label1}={completions1['tokens'][batch_idx, pos_idx].item()}, "
                  f"{label2}={completions2['tokens'][batch_idx, pos_idx].item()}")
        
        # Show token statistics
        print(f"\n   Token Statistics:")
        print(f"     {label1} - unique tokens: {completions1['tokens'].unique().numel()}, "
              f"min: {completions1['tokens'].min().item()}, max: {completions1['tokens'].max().item()}")
        print(f"     {label2} - unique tokens: {completions2['tokens'].unique().numel()}, "
              f"min: {completions2['tokens'].min().item()}, max: {completions2['tokens'].max().item()}")
        
        assert False, "Token tensors differ"
    else:
        print(f"   ✓ All tokens match")
    
    # Check text equality
    print(f"\n3. Text Output Comparison:")
    print(f"   Number of texts: {label1}={len(completions1['texts'])}, {label2}={len(completions2['texts'])}")
    
    texts_equal = completions1['texts'] == completions2['texts']
    print(f"   Texts equal: {texts_equal}")
    
    if not texts_equal:
        print(f"   ❌ TEXT OUTPUTS DIFFER!")
        print(f"\n   Detailed text comparison:")
        for i, (text1, text2) in enumerate(zip(completions1['texts'], completions2['texts'])):
            if text1 != text2:
                print(f"\n   Completion {i} differs:")
                print(f"     {label1} length: {len(text1)}")
                print(f"     {label2} length: {len(text2)}")
                print(f"\n     {label1} text (first 200 chars):")
                print(f"     {repr(text1[:200])}")
                print(f"\n     {label2} text (first 200 chars):")
                print(f"     {repr(text2[:200])}")
                
                # Find first difference
                min_len = min(len(text1), len(text2))
                for j in range(min_len):
                    if text1[j] != text2[j]:
                        print(f"\n     First character difference at position {j}:")
                        print(f"       {label1}: {repr(text1[max(0, j-10):j+10])}")
                        print(f"       {label2}: {repr(text2[max(0, j-10):j+10])}")
                        break
        
        assert False, "Text outputs differ"
    else:
        print(f"   ✓ All texts match")
    
    # Check exit layer count
    print(f"\n4. Exit Layer Count Comparison:")
    print(f"   {label1}: {len(exit_info1['prescribed_exit_layers'])} completions")
    print(f"   {label2}: {len(exit_info2['prescribed_exit_layers'])} completions")
    
    if len(exit_info1['prescribed_exit_layers']) != len(exit_info2['prescribed_exit_layers']):
        print(f"   ❌ EXIT LAYER COUNTS DIFFER!")
        assert False, "Exit layer count differs"
    else:
        print(f"   ✓ Exit layer counts match")
    
    # Check exit layer values
    print(f"\n5. Exit Layer Values Comparison:")
    all_layers_match = True
    
    for i, (layers1, layers2) in enumerate(zip(
        exit_info1['prescribed_exit_layers'],
        exit_info2['prescribed_exit_layers']
    )):
        layers_match = len(layers1) == len(layers2) and torch.equal(layers1, layers2)
        
        if not layers_match:
            all_layers_match = False
            print(f"\n   Completion {i}:")
            print(f"     {label1} exit layers length: {len(layers1)}")
            print(f"     {label2} exit layers length: {len(layers2)}")
            
            if len(layers1) != len(layers2):
                print(f"     ❌ LENGTHS DIFFER!")
            else:
                # Show differences in values
                diff_mask = layers1 != layers2
                num_diffs = diff_mask.sum().item()
                print(f"     ❌ VALUES DIFFER at {num_diffs} positions")
                
                # Show first few values for comparison
                print(f"\n     First 20 exit layer values:")
                print(f"       {label1}: {layers1[:20].tolist()}")
                print(f"       {label2}: {layers2[:20].tolist()}")
                
                # Show statistics
                print(f"\n     Statistics:")
                finite1 = torch.isfinite(layers1)
                finite2 = torch.isfinite(layers2)
                print(f"       {label1} - finite: {finite1.sum().item()}, inf: {(~finite1).sum().item()}")
                print(f"       {label2} - finite: {finite2.sum().item()}, inf: {(~finite2).sum().item()}")
                
                if finite1.any():
                    print(f"       {label1} - finite values mean: {layers1[finite1].float().mean().item():.2f}")
                if finite2.any():
                    print(f"       {label2} - finite values mean: {layers2[finite2].float().mean().item():.2f}")
    
    if all_layers_match:
        print(f"   ✓ All exit layer values match for all completions")
    else:
        print(f"\n   ❌ SOME EXIT LAYERS DIFFER!")
        assert False, "Exit layer values differ for some completions"
    
    print(f"\n{'=' * 80}")
    print(f"✓ ALL CHECKS PASSED")
    print(f"{'=' * 80}\n")


# ============================================================================
# Test Cases
# ============================================================================

def test_attention_mask_exit_layer_consistency(model_setup, system_prompt, test_prompts):
    """
    Lightweight test for attention mask vs exit layer length consistency.
    
    This test checks the critical assertion from the notebook:
    generated_attention_mask.sum(-1).tolist() == [len(item) for item in exit_info['prescribed_exit_layers']]
    
    This assertion works for sequential generation but fails for batched generation.
    """
    from early_exit.rl_utils import get_input_prompt_length, create_attention_mask_from_tokens
    
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 50
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = [test_prompts[0]]
    k = 2
    
    print(f"\n{'=' * 80}")
    print(f"LIGHTWEIGHT TEST: Attention Mask vs Exit Layer Length Consistency")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  - Prompts: {len(prompts)}")
    print(f"  - K: {k}")
    print(f"  - Max new tokens: {config['generation']['max_new_tokens']}")
    print(f"  - Prompt: {prompts[0]}")
    
    # Get input prompt length (needed for slicing)
    input_prompt_length = get_input_prompt_length(
        model_setup['tokenizer'], 
        prompts[0], 
        system_prompt=system_prompt
    )
    print(f"\n  - Input prompt length: {input_prompt_length} tokens")
    
    # Test Sequential
    print(f"\n--- Testing Sequential (generate_k_completions) ---")
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    print(f"Sequential Results:")
    print(f"  - Tokens shape: {completions_seq['tokens'].shape}")
    print(f"  - Number of completions: {len(completions_seq['texts'])}")
    print(f"  - Number of exit layer sequences: {len(exit_info_seq['prescribed_exit_layers'])}")
    
    # Create attention mask and slice off prompt
    generated_attention_mask_seq = create_attention_mask_from_tokens(
        completions_seq['tokens'], 
        model_setup['tokenizer'].pad_token_id
    )[:, input_prompt_length:]
    
    print(f"\n  Attention mask (after prompt slice):")
    print(f"    - Shape: {generated_attention_mask_seq.shape}")
    print(f"    - Sum per completion: {generated_attention_mask_seq.sum(-1).tolist()}")
    
    exit_layer_lengths_seq = [len(item) for item in exit_info_seq['prescribed_exit_layers']]
    print(f"\n  Exit layer lengths: {exit_layer_lengths_seq}")
    
    # Critical assertion
    try:
        assert generated_attention_mask_seq.sum(-1).tolist() == exit_layer_lengths_seq
        print(f"  ✓ SEQUENTIAL ASSERTION PASSED")
    except AssertionError as e:
        print(f"  ❌ SEQUENTIAL ASSERTION FAILED!")
        print(f"     Mask sums: {generated_attention_mask_seq.sum(-1).tolist()}")
        print(f"     Exit lengths: {exit_layer_lengths_seq}")
        raise
    
    # Test Batched
    print(f"\n--- Testing Batched (generate_k_completions_batched) ---")
    with torch.no_grad():
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    print(f"Batched Results:")
    print(f"  - Tokens shape: {completions_batch['tokens'].shape}")
    print(f"  - Number of completions: {len(completions_batch['texts'])}")
    print(f"  - Number of exit layer sequences: {len(exit_info_batch['prescribed_exit_layers'])}")
    
    # Create attention mask and slice off prompt
    generated_attention_mask_batch = create_attention_mask_from_tokens(
        completions_batch['tokens'], 
        model_setup['tokenizer'].pad_token_id
    )[:, input_prompt_length:]
    
    print(f"\n  Attention mask (after prompt slice):")
    print(f"    - Shape: {generated_attention_mask_batch.shape}")
    print(f"    - Sum per completion: {generated_attention_mask_batch.sum(-1).tolist()}")
    
    exit_layer_lengths_batch = [len(item) for item in exit_info_batch['prescribed_exit_layers']]
    print(f"\n  Exit layer lengths: {exit_layer_lengths_batch}")
    
    # Critical assertion
    try:
        assert generated_attention_mask_batch.sum(-1).tolist() == exit_layer_lengths_batch
        print(f"  ✓ BATCHED ASSERTION PASSED")
    except AssertionError as e:
        print(f"  ❌ BATCHED ASSERTION FAILED!")
        print(f"     Mask sums: {generated_attention_mask_batch.sum(-1).tolist()}")
        print(f"     Exit lengths: {exit_layer_lengths_batch}")
        
        # Additional diagnostics
        print(f"\n  Detailed Diagnostics:")
        for i in range(len(exit_layer_lengths_batch)):
            mask_sum = generated_attention_mask_batch[i].sum().item()
            exit_len = exit_layer_lengths_batch[i]
            diff = mask_sum - exit_len
            print(f"    Completion {i}: mask_sum={mask_sum}, exit_len={exit_len}, diff={diff}")
        
        # Check actual tokens
        print(f"\n  Token Analysis (first completion):")
        tokens_first = completions_batch['tokens'][0]
        print(f"    Total tokens: {len(tokens_first)}")
        print(f"    Prompt tokens: {input_prompt_length}")
        print(f"    Generated tokens: {len(tokens_first) - input_prompt_length}")
        
        # Check for pad tokens
        pad_token_id = model_setup['tokenizer'].pad_token_id
        num_pads = (tokens_first == pad_token_id).sum().item()
        print(f"    Pad tokens (total): {num_pads}")
        
        # Find first pad in generated portion
        gen_tokens = tokens_first[input_prompt_length:]
        pad_positions = (gen_tokens == pad_token_id).nonzero(as_tuple=False)
        if len(pad_positions) > 0:
            first_pad = pad_positions[0].item()
            print(f"    First pad in generated: position {first_pad}")
        else:
            print(f"    No pads in generated portion")
        
        raise
    
    print(f"\n{'=' * 80}")
    print(f"✓ BOTH SEQUENTIAL AND BATCHED PASS THE CONSISTENCY CHECK")
    print(f"{'=' * 80}\n")


def test_single_prompt_k2(model_setup, system_prompt, test_prompts):
    """Test with 1 prompt, k=2, short generation."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 50
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = [test_prompts[0]]
    k = 2
    
    print(f"\nTesting: {len(prompts)} prompt(s), k={k}, max_tokens={config['generation']['max_new_tokens']}")
    print(f"Prompt: {prompts[0]}")
    
    # Generate with both methods
    with torch.no_grad():
        print(f"\n--- Generating with generate_k_completions (sequential) ---")
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        print_generation_summary(completions_seq, exit_info_seq, "Sequential")
        
        print(f"\n--- Generating with generate_k_completions_batched ---")
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        print_generation_summary(completions_batch, exit_info_batch, "Batched")
    
    # Assert equality
    assert_completions_equal(completions_seq, completions_batch, exit_info_seq, exit_info_batch)
    
    # Verify expected counts
    expected_total = len(prompts) * k
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    print(f"✓ Generated {expected_total} completions successfully")


def test_multiple_prompts_k4(model_setup, system_prompt, test_prompts):
    """Test with 2 prompts, k=4, medium generation."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 100
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = test_prompts[:2]
    k = 4
    
    print(f"\nTesting: {len(prompts)} prompt(s), k={k}, max_tokens={config['generation']['max_new_tokens']}")
    
    # Generate with both methods
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # Assert equality
    assert_completions_equal(completions_seq, completions_batch, exit_info_seq, exit_info_batch)
    
    # Verify expected counts (2 prompts × 4 k = 8)
    expected_total = len(prompts) * k
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    print(f"✓ Generated {expected_total} completions successfully")


def test_many_prompts_k8(model_setup, system_prompt, test_prompts):
    """Test with 3 prompts, k=8, longer generation."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 200
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = test_prompts
    k = 8
    
    print(f"\nTesting: {len(prompts)} prompt(s), k={k}, max_tokens={config['generation']['max_new_tokens']}")
    
    # Generate with both methods
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # Assert equality
    assert_completions_equal(completions_seq, completions_batch, exit_info_seq, exit_info_batch)
    
    # Verify expected counts (3 prompts × 8 k = 24)
    expected_total = len(prompts) * k
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    print(f"✓ Generated {expected_total} completions successfully")


def test_with_temperature(model_setup, system_prompt, test_prompts):
    """Test with temperature=1.0 sampling (shape consistency only)."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 50
    config['generation']['temperature'] = 1.0
    config['generation']['do_sample'] = True
    
    prompts = [test_prompts[0]]
    k = 2
    
    print(f"\nTesting with sampling: {len(prompts)} prompt(s), k={k}, temperature=1.0")
    
    # Generate with both methods
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # With sampling, outputs may differ between runs
    # Only test shape consistency and format validity
    expected_total = len(prompts) * k
    
    # Check shapes are consistent
    assert completions_seq['tokens'].shape[0] == expected_total
    assert completions_batch['tokens'].shape[0] == expected_total
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    # Check exit layer counts
    assert len(exit_info_seq['prescribed_exit_layers']) == expected_total
    assert len(exit_info_batch['prescribed_exit_layers']) == expected_total
    
    print(f"✓ Shape consistency verified for {expected_total} completions")


def test_edge_case_k1(model_setup, system_prompt, test_prompts):
    """Test edge case with k=1."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 50
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = [test_prompts[0]]
    k = 1
    
    print(f"\nTesting edge case: {len(prompts)} prompt(s), k={k}")
    
    # Generate with both methods
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # Assert equality
    assert_completions_equal(completions_seq, completions_batch, exit_info_seq, exit_info_batch)
    
    # Verify expected counts
    expected_total = len(prompts) * k
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    print(f"✓ k=1 edge case works correctly")


def test_exit_layer_consistency(model_setup, system_prompt, test_prompts):
    """Test that exit layer information is consistent."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = 100
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = [test_prompts[0]]
    k = 2
    
    print(f"\nTesting exit layer consistency: {len(prompts)} prompt(s), k={k}")
    
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # Check exit layers for each completion
    for i in range(len(exit_info_seq['prescribed_exit_layers'])):
        seq_layers = exit_info_seq['prescribed_exit_layers'][i]
        batch_layers = exit_info_batch['prescribed_exit_layers'][i]
        
        # Should have same length
        assert len(seq_layers) == len(batch_layers), \
            f"Exit layer length differs for completion {i}"
        
        # Should have same values
        assert torch.equal(seq_layers, batch_layers), \
            f"Exit layer values differ for completion {i}"
        
        # Verify layers are valid (not all inf, contain some actual layer values)
        finite_seq = torch.isfinite(seq_layers).sum()
        finite_batch = torch.isfinite(batch_layers).sum()
        assert finite_seq == finite_batch, \
            f"Number of finite exit layers differs for completion {i}"
        
        assert finite_seq > 0, \
            f"Completion {i} has no finite exit layers (all inf)"
    
    print(f"✓ Exit layer consistency verified for {len(exit_info_seq['prescribed_exit_layers'])} completions")


@pytest.mark.parametrize("num_prompts,k,max_tokens", [
    (1, 2, 50),
    (1, 4, 100),
    (2, 2, 50),
    (2, 4, 100),
    (3, 2, 75),
])
def test_various_configurations(model_setup, system_prompt, test_prompts,
                               num_prompts, k, max_tokens):
    """Parameterized test for various configurations."""
    config = deepcopy(model_setup['config'])
    config['generation']['max_new_tokens'] = max_tokens
    config['generation']['do_sample'] = False
    config['generation']['temperature'] = 0
    
    prompts = test_prompts[:num_prompts]
    
    print(f"\nTesting configuration: prompts={num_prompts}, k={k}, max_tokens={max_tokens}")
    
    with torch.no_grad():
        completions_seq, exit_info_seq = generate_k_completions(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
        
        completions_batch, exit_info_batch = generate_k_completions_batched(
            model_setup['model'], prompts, k,
            model_setup['tokenizer'], config,
            model_setup['device'], system_prompt
        )
    
    # Standard assertions
    assert completions_seq['tokens'].shape == completions_batch['tokens'].shape
    assert torch.equal(completions_seq['tokens'], completions_batch['tokens'])
    assert completions_seq['texts'] == completions_batch['texts']
    
    # Verify expected total
    expected_total = num_prompts * k
    assert len(completions_seq['texts']) == expected_total
    assert len(completions_batch['texts']) == expected_total
    
    print(f"✓ Configuration verified: {expected_total} completions match")


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

