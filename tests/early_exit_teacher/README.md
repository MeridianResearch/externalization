# Early Exit Visualization

The [visualization](https://htmlpreview.github.io/?https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/visualizations/early_exit_comparison.html) and [code](https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/modeling_exit.py) consists of three modes:

1. Normal generation without early exiting
2. Early exiting without freezing
3. Early exiting after committing to early exiting

### Random facts that can be useful later
1. I encountered an error in one of the runs because I didn’t clone the student and teacher caches separately, which led to the repeated token issue. A similar problem might be occurring in `patched_attention_forward()`, where we’re not cloning the `past_key_values`. Turns out the argument sent to `past_key_values` gets updated in-place.
2. `past_key_values`: 
    ```
    Tuple[ 
        Tuple[Tensor, Tensor],  # Layer 0: (past_key, past_value)
        Tuple[Tensor, Tensor],  # Layer 1
        ...
        Tuple[Tensor, Tensor]   # Layer N
    ]
    ```

### TODO: Check why the outputs from the notebook is not the same as the code here. Is it just because of the sampling of early exit?