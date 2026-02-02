import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ast import literal_eval

import torch
import pandas as pd
import numpy as np
import html
import matplotlib.colors as mcolors
from IPython.display import HTML, display
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import html
import matplotlib.colors as mcolors
from IPython.display import HTML, display
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import html
import matplotlib.colors as mcolors
from IPython.display import HTML, display
from transformers import AutoTokenizer


def safe_decode_tokens(tokenizer, tokens):
    """Safely decode tokens to strings."""
    try:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Decoding failed: {e}")
        return ""


def visualize_tokens_by_exit_layer(token_strings, exit_layers, early_exit_layer_idxs=None, 
                                  title="Token Early Exit Visualization", prompt="", 
                                  save_html=None, limit=None, show_prompt=True, computation_saved=None):
    """
    Visualize tokens colored by their early exit layers.
    Expects token_strings to be a LIST of strings, not a single string.
    """
    
    # Slice inputs if limit is set
    if limit is not None:
        token_strings = token_strings[:limit]
        exit_layers = exit_layers[:limit]

    # Get all unique layers to determine the range
    unique_layers = sorted(set(exit_layers))
    if early_exit_layer_idxs is not None:
        all_layers = list(early_exit_layer_idxs) + [max(exit_layers) if exit_layers else 36]
        unique_layers = sorted(set(all_layers))
    
    # Simplified Color Setup for layers 20-36
    # Use a simple gradient from dark to light
    custom_hex_colors = [
        '#6E4C4B',  # Layer 20 (Dark Brown/Red)
        '#975654',
        '#C08872',
        '#D6B886',
        '#EBE3D9',
        '#FAFAFA'   # Layer 36 (Almost White)
    ]
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_early_exit", custom_hex_colors)
    
    # Normalize between 20 and 36 (or the actual range)
    min_layer = min(unique_layers) if unique_layers else 20
    max_layer = max(unique_layers) if unique_layers else 36
    norm = mcolors.Normalize(vmin=min_layer, vmax=max_layer)
    
    layer_colors = {}
    for layer in unique_layers:
        rgba = cmap(norm(layer))
        layer_colors[layer] = mcolors.to_hex(rgba)

    # Build HTML
    title_with_computation = title
    if computation_saved is not None:
        title_with_computation = f"{title} (Computation Saved: {computation_saved})"
    
    html_content = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; padding: 20px; 
                background-color: #f9f9f9; border-radius: 10px;">
        <h3 style="text-align: center; color: #333; margin-bottom: 20px;">{title_with_computation}</h3>
    """
    
    if prompt and show_prompt:
        html_content += f"""
        <div style="margin: 15px 0; padding: 12px; background-color: #fff3cd; 
                    border-left: 4px solid #D6B886; border-radius: 5px;">
            <strong style="color: #000;">Prompt:</strong> 
            <span style="color: #000;">{html.escape(prompt)}</span>
        </div>
        """
    
    # No legend needed since we're only using layers 20-36
    html_content += """
        <div style="line-height: 2.5; word-wrap: break-word; padding: 15px; 
                    background-color: #fff; border-radius: 5px; border: 1px solid #ddd;">
    """
    
    # Tokens Loop
    for token, exit_layer in zip(token_strings, exit_layers):
        color = layer_colors.get(exit_layer, "#FFFFFF")
        
        # Escape HTML characters in the token string
        token_display = html.escape(token, quote=False)
        
        # Visualize whitespace slightly
        token_display = token_display.replace('\n', '<span style="opacity:0.3">\\n</span>')
        token_display = token_display.replace('\t', '<span style="opacity:0.3">\\t</span>')
        
        # Determine text color based on background brightness
        hex_c = color.lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "white" if brightness < 150 else "black"
        
        html_content += f"""<span style="display: inline-block; padding: 4px 8px; margin: 2px; 
                                      border-radius: 4px; border: 1px solid #ccc; 
                                      font-family: monospace; font-size: 13px; 
                                      background-color: {color}; color: {text_color}; 
                                      font-weight: bold; max-width: 200px; 
                                      overflow-wrap: break-word; vertical-align: middle;" 
                                      title="Layer {exit_layer}">{token_display}</span>"""
    
    html_content += "</div>"
        
    layer_counts = {l: exit_layers.count(l) for l in unique_layers if exit_layers.count(l) > 0}
    stats_items = [f"L{l}: {c}" for l, c in layer_counts.items()]
    stats_text = " &nbsp;|&nbsp; ".join(stats_items)
    
    html_content += f"""
        <div style="margin-top: 15px; padding: 10px; background-color: #f0f0f0; 
                    border-radius: 5px; font-family: monospace; font-size: 13px; color: #333;">
            <strong>Showing {len(token_strings)} tokens</strong> &nbsp;|&nbsp; {stats_text}
        </div>
    </div>
    """
    
    if save_html:
        full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_content}</body></html>"
        with open(save_html, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"HTML visualization saved to: {save_html}")
        return html_content 
    else:
        return html_content


def parse_token_data(token_ids_str, tokens_text_str):
    """Parse token IDs and token text from CSV strings."""
    # Parse token IDs
    token_ids = eval(token_ids_str)
    
    # Parse token text
    tokens_text = eval(tokens_text_str)
    
    return token_ids, tokens_text


def parse_exit_layers(exit_layers_str):
    """Parse exit layers from CSV string."""
    exit_layers = eval(exit_layers_str)
    # Convert 'inf' to 36
    layers = [36 if item == 'inf' or item == -1 else int(item) for item in exit_layers]
    return layers


def create_side_by_side_visualization(prompt, tokens_pre, layers_pre, tokens_post, layers_post, 
                                     early_exit_layer_idxs=None, row_idx=0, 
                                     computation_saved_pre=None, computation_saved_post=None):
    """Create side-by-side visualization for pre and post RL."""
    
    # Get unique layers from both pre and post to build the legend
    all_layers = sorted(set(layers_pre + layers_post))
    
    # Create color mapping (same as in visualize function)
    custom_hex_colors = [
        '#6E4C4B',  # Layer 20 (Dark Brown/Red)
        '#975654',
        '#C08872',
        '#D6B886',
        '#EBE3D9',
        '#FAFAFA'   # Layer 36 (Almost White)
    ]
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_early_exit", custom_hex_colors)
    min_layer = min(all_layers) if all_layers else 20
    max_layer = max(all_layers) if all_layers else 36
    norm = mcolors.Normalize(vmin=min_layer, vmax=max_layer)
    
    layer_colors = {}
    for layer in all_layers:
        rgba = cmap(norm(layer))
        layer_colors[layer] = mcolors.to_hex(rgba)
    
    # Create wrapper HTML with prompt at the top
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='UTF-8'>
        <style>
            body {{ margin: 0; padding: 20px; background-color: white; }}
            .prompt-section {{
                margin: 20px auto;
                max-width: 1800px;
                padding: 15px;
                background-color: #fff3cd;
                border-left: 4px solid #D6B886;
                border-radius: 5px;
            }}
            .legend-section {{
                margin: 20px auto;
                max-width: 1800px;
                display: flex;
                justify-content: center;
                gap: 15px;
                padding: 15px;
                background-color: #fff;
                border-radius: 5px;
                flex-wrap: wrap;
                border: 1px solid #ddd;
            }}
            .comparison-container {{
                display: flex;
                gap: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }}
            .visualization-half {{
                flex: 1;
            }}
        </style>
    </head>
    <body>
        <div class="prompt-section">
            <strong style="color: #000;">Prompt:</strong> 
            <span style="color: #000;">{html.escape(prompt)}</span>
        </div>
        <div class="legend-section">
    """
    
    # Add legend items
    for layer in all_layers:
        color = layer_colors.get(layer, "#FFFFFF")
        html_content += f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 25px; height: 15px; background-color: {color}; 
                            border: 1px solid #999; border-radius: 3px;"></div>
                <span style="font-size: 14px; color: #000; font-weight: 500;">Layer {layer}</span>
            </div>
        """
    
    html_content += """
        </div>
        <div class="comparison-container">
            <div class="visualization-half">
    """
    
    # Pre-RL visualization
    pre_html = visualize_tokens_by_exit_layer(
        tokens_pre,
        layers_pre,
        early_exit_layer_idxs,
        title="Pre-RL",
        show_prompt=False,
        computation_saved=computation_saved_pre
    )
    
    html_content += pre_html
    html_content += """
            </div>
            <div class="visualization-half">
    """
    
    # Post-RL visualization
    post_html = visualize_tokens_by_exit_layer(
        tokens_post,
        layers_post,
        early_exit_layer_idxs,
        title="Post-RL",
        show_prompt=False,
        computation_saved=computation_saved_post
    )
    
    html_content += post_html
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def main():
    # Load CSV
    df = pd.read_csv('early_exit_results.csv')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Process each row
    for idx, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing Row {idx + 1}/{len(df)}")
        print(f"{'='*80}")
        
        # Extract data
        prompt = row['prompt']
        
        # Pre-RL data
        token_ids_pre, tokens_text_pre = parse_token_data(
            row['token_ids_preRL'], 
            row['tokens_text_preRL']
        )
        layers_pre = parse_exit_layers(row['exit_layers_preRL'])
        computation_saved_pre = row['computation_saved_preRL']
        
        # Post-RL data
        token_ids_post, tokens_text_post = parse_token_data(
            row['token_ids_postRL'], 
            row['tokens_text_postRL']
        )
        layers_post = parse_exit_layers(row['exit_layers_postRL'])
        computation_saved_post = row['computation_saved_postRL']
        
        print(f"Pre-RL: {len(tokens_text_pre)} tokens, Computation Saved: {computation_saved_pre}")
        print(f"Post-RL: {len(tokens_text_post)} tokens, Computation Saved: {computation_saved_post}")
        
        # Create side-by-side visualization
        html_output = create_side_by_side_visualization(
            prompt=prompt,
            tokens_pre=tokens_text_pre,
            layers_pre=layers_pre,
            tokens_post=tokens_text_post,
            layers_post=layers_post,
            early_exit_layer_idxs=[20, 25, 30, 35],  # Only layers 20+
            row_idx=idx,
            computation_saved_pre=computation_saved_pre,
            computation_saved_post=computation_saved_post
        )
        
        # Save to file
        output_file = f'visualization_row_{idx + 1}.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"✅ Saved: {output_file}")
        
        # Try to save as PNG
        try:
            from html2image import Html2Image
            hti = Html2Image(output_path='./')
            
            png_file = f'visualization_row_{idx + 1}.png'
            hti.screenshot(html_str=html_output, save_as=png_file, size=(1800, 1200))
            print(f"✅ PNG saved: {png_file}")
        except ImportError:
            print("⚠️ html2image not installed. Install with: pip install html2image")
        except Exception as e:
            print(f"⚠️ Could not save PNG: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ All visualizations complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()