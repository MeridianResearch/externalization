# Early Exit Teacher Module

This module implements an early exit mechanism for transformer models, allowing the model to dynamically decide when to stop processing at intermediate layers rather than always going through all layers.

## Overview

The early exit mechanism enables efficient inference by allowing models to exit at earlier layers when confident about predictions. This reduces computational costs while maintaining accuracy.

## Execution Modes

The module supports three modes of operation:

1. **Normal Generation**: Standard full-depth generation without early exits
2. **Unfrozen Mode**: Early exit without freezing the residual stream 
3. **Frozen Residual Mode**: Early exit with frozen residual stream after exiting

## Core Components

### `early_exit_predictions.py`
**Main inference and generation module**

Key classes and functions:

- **`EarlyExitGenerator` & `KLExitGenerator`**: Core generation classes
  - `EarlyExitGenerator`: Base class providing fundamental early exit functionality
  - `KLExitGenerator`: Extended class that adds KL-divergence based exit decisions
  - Computes KL divergence between intermediate and final layer outputs
  - Uses configurable KL strength factors (0.5, 1.0, 2.0, 4.0) to control exit aggressiveness
  - Implements the three generation modes (normal, unfrozen, frozen_residual)

- **`PredictionObject`**: Maintains generation state
  - Tracks generated tokens, chosen exit layers, and logits
  - Manages key-value cache for autoregressive generation
  - Handles cache manipulation for different exit modes

- **Utility functions**:
  - `load_default_model_and_tokenizer()`: Loads model with early exit configuration
  - `get_early_exit_indices()`: Identifies which layers can serve as exit points
  - `format_and_tokenize_input()`: Prepares prompts for model input
  - `evaluate_response()`: Analyzes generation quality and exit statistics

### `visualization.py`
**Visualization tools for analyzing early exit behavior**

Creates interactive HTML dashboards showing:

- **Token-level visualization**: Color-coded tokens based on exit layer (blue=early, red=late)
- **Statistical analysis**: Exit rates, layer distributions, and performance metrics
- **Multi-prompt comparison**: Aggregate statistics across different prompts and configurations

Key functions:
- `generate_multi_prompt_html_visualization()`: Creates hierarchical dashboard
- `save_results_json()` / `load_results_json()`: Data persistence
- `create_html_visualization()`: Single-prompt visualization

### `run_early_predictions.py`
**Main experiment runner**

Orchestrates experiments by:
1. Loading model and test prompts
2. Testing all combinations of modes and KL factors
3. Collecting generation results and exit statistics
4. Saving results to JSON and generating HTML visualizations

The script automatically tests multiple prompts across all configurations and produces comprehensive analysis outputs.

## How Early Exit Works

The module uses **KL divergence** between intermediate and final layer outputs to determine when to exit:

- **Low KL divergence** = High confidence → Safe to exit early
- **High KL divergence** = Low confidence → Continue to deeper layers
- **KL factor parameter** controls sensitivity (higher = more conservative)

Exit decisions are made at predetermined "exitable" layers (e.g., every 5th layer) using a stick-breaking probability process.

## Usage Example

```python
# Load model with early exit
model, tokenizer = load_default_model_and_tokenizer()

# Initialize generator with desired mode
generator = KLExitGenerator(
    model=model,
    tokenizer=tokenizer,
    mode='frozen_residual',  # or 'normal', 'unfrozen'
    exitable_layers=get_early_exit_indices(model)
)

# Generate with early exit
tokens, exit_layers, prediction = generator.generate(
    inputs=formatted_input,
    kl_strength=1.0,  # KL factor for exit sensitivity
    max_new_tokens=100
)

# Analyze results
results = generator.evaluate_response(tokens, exit_layers, inputs)
```

## Output Files

- **`early_exit_analysis.json`**: Complete experiment results with tokens, exit layers, and statistics
- **`early_exit_analysis.html`**: Interactive visualization dashboard
- Model outputs include early exit rates, layer distributions, and performance metrics