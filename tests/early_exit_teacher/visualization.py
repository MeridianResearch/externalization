import os
import html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
import string
import json
from typing import Dict, List, Any
import pickle
from datetime import datetime


def create_html_visualization(all_results, early_exit_layer_idxs, test_prompts,
                             output_path='tests/prompt_based_kl_output.html',
                             title='Early Exit Behavior Visualization'):
    """
    Create an HTML file with visualization of early exit behavior across different modes and KL strengths.
    
    Args:
        all_results: List of lists structure where:
                    all_results[prompt_idx] = [
                        {
                            'mode': str,
                            'kl_strength': float or None,
                            'data': (tokens, exit_layers, text, kl_divs)
                        }, ...
                    ]
        early_exit_layer_idxs: Tensor of available early exit layers
        test_prompts: List of test prompts
        output_path: Path to save the HTML file
        title: Custom title for the visualization
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create color mapping - get the actual final layer index from the model
    # Find the maximum layer index from the data to determine final layer
    max_layer = 0
    for prompt_results in all_results:
        for result_item in prompt_results:
            if result_item:  # Check if result_item exists
                _, exit_layers, _, _ = result_item['data']
                if exit_layers:
                    max_layer = max(max_layer, max(exit_layers))
    
    all_layers = list(early_exit_layer_idxs.numpy()) + [max_layer]  # Use actual final layer
    # Remove duplicates and sort
    all_layers = sorted(list(set(all_layers)))
    cmap = plt.colormaps.get_cmap('coolwarm_r')  # Blue to red
    norm = mcolors.Normalize(vmin=0, vmax=len(all_layers)-1)
    
    # Generate colors for each layer
    layer_colors = {}
    for i, layer in enumerate(all_layers):
        color = cmap(norm(i))
        # Convert to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255)
        )
        layer_colors[layer] = hex_color
    
    # Escape the title for HTML and use it in both the page title and header
    escaped_title = html.escape(title)
    
    # Start building HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escaped_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            color: #666;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 30px;
            height: 20px;
            border: 1px solid #333;
            border-radius: 3px;
        }}
        .tokens-container {{
            margin: 15px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
            line-height: 2.2;
            word-wrap: break-word;
        }}
        .token {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            border: 1px solid #666;
            font-family: monospace;
            font-size: 14px;
            color: white;
            font-weight: bold;
            max-width: 200px;
            overflow-wrap: break-word;
            vertical-align: middle;
            cursor: pointer;
            position: relative;
        }}
        .token:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            z-index: 10;
        }}
        .token .tooltip {{
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 12px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            font-size: 12px;
            font-weight: normal;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .token .tooltip::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }}
        .token:hover .tooltip {{
            visibility: visible;
        }}
        .stats {{
            margin: 15px 0;
            padding: 10px;
            background-color: #e8f4fd;
            border-radius: 5px;
            font-family: monospace;
            font-size: 13px;
        }}
        .summary-table {{
            margin: 20px 0;
            border-collapse: collapse;
            width: 100%;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        .summary-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .prompt-section {{
            margin-bottom: 50px;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background-color: #fdfdfd;
        }}
        .mode-section {{
            margin-bottom: 30px;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
        }}
        .no-data {{
            font-style: italic;
            color: #999;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{escaped_title}</h1>
        <p style="text-align: center; color: #666; font-style: italic; margin-bottom: 30px;">
            Token colors indicate the exit layer used for its generation.
        </p>
        <!-- Color Legend -->
        <div class="legend">
"""
    
    # Add legend items
    for layer in all_layers:
        # Determine if this is the final layer (highest layer index)
        layer_name = f"Layer {layer}" if layer != max(all_layers) else "Final Layer"
        color = layer_colors[layer]
        # Determine text color based on background brightness
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "white" if brightness < 128 else "black"
        
        html_content += f"""
            <div class="legend-item">
                <div class="legend-box" style="background-color: {color};"></div>
                <span>{layer_name}</span>
            </div>
"""
    
    html_content += """
        </div>
"""
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(test_prompts):
        escaped_prompt = html.escape(prompt)
        html_content += f"""
        <div class="prompt-section">
            <h2>Prompt {prompt_idx + 1}: "{escaped_prompt}"</h2>
"""
        
        # Check if we have results for this prompt
        if prompt_idx >= len(all_results) or not all_results[prompt_idx]:
            html_content += """
            <div class="no-data">No data available for this prompt</div>
        </div>
"""
            continue
        
        # Process each mode/KL combination for this prompt
        all_stats = {}
        
        # First pass: count occurrences of each base configuration
        base_config_counts = {}
        for result_item in all_results[prompt_idx]:
            mode = result_item['mode']
            kl_strength = result_item['kl_strength']
            base_config = f"{mode} (KL: {kl_strength})" if kl_strength is not None else mode
            base_config_counts[base_config] = base_config_counts.get(base_config, 0) + 1
        
        # Second pass: generate display names with generation numbers
        config_counters = {}
        for result_item in all_results[prompt_idx]:
            mode = result_item['mode']
            kl_strength = result_item['kl_strength']
            token_strings, exit_layers, _, kl_divs = result_item['data']
            
            # Create base display name
            base_config_name = f"{mode} (KL: {kl_strength})" if kl_strength is not None else mode
            
            # Generate unique display name
            if base_config_counts[base_config_name] > 1:
                # Multiple occurrences - add generation number
                config_counters[base_config_name] = config_counters.get(base_config_name, 0) + 1
                config_name = f"{base_config_name} - Sample {config_counters[base_config_name]}"
            else:
                # Single occurrence - no generation number needed
                config_name = base_config_name
            
            html_content += f"""
            <div class="mode-section">
                <h3>{config_name}</h3>
"""
            
            # Display tokens
            html_content += """
                <div class="tokens-container">
"""
            
            for i, (token, exit_layer) in enumerate(zip(token_strings, exit_layers)):
                color = layer_colors[exit_layer]
                # Escape special characters in token and handle unicode properly
                token_display = html.escape(token, quote=False)
                # Replace common whitespace and control characters with visible representations
                token_display = token_display.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                # Handle other special characters
                token_display = token_display.replace('\u00a0', '[NBSP]')  # Non-breaking space
                token_display = token_display.replace('\ufeff', '[BOM]')   # Byte order mark
                # Replace any remaining non-printable characters
                token_display = ''.join(char if char.isprintable() or char in ' \n\t' else f'[U+{ord(char):04X}]' for char in token_display)
                
                # Determine text color based on background
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = "white" if brightness < 128 else "black"
                
                # Create tooltip content
                layer_display = "Final Layer" if exit_layer == max(all_layers) else f"Layer {exit_layer}"
                tooltip_content = f"Exit Layer: {layer_display}"
                
                if i < len(kl_divs) and kl_divs[i] is not None:
                    # Handle 1D tensor case - kl_divs[i] is a tensor of shape [num_layers]
                    if hasattr(kl_divs[i], 'shape') and len(kl_divs[i].shape) == 1:
                        # kl_divs[i] is already a 1D tensor with KL values for each layer
                        tooltip_lines = [f"Exit Layer: {layer_display}"]
                        tooltip_lines.append("Exit Probs:")
                        for j, kl_val in enumerate(kl_divs[i]):
                            if j < len(early_exit_layer_idxs):
                                layer_idx = early_exit_layer_idxs[j].item()
                                tooltip_lines.append(f"Layer {layer_idx}: {kl_val.item():.2f}")
                        tooltip_content = "<br>".join(tooltip_lines)
                    elif hasattr(kl_divs[i], 'item'):
                        # Single scalar tensor
                        tooltip_content += f"<br>Exit Probs: {kl_divs[i].item():.2f}"
                    else:
                        # Fallback for other cases
                        tooltip_content += f"<br>Exit Probs: {float(kl_divs[i]):.2f}"
                
                html_content += f"""<span class="token" style="background-color: {color}; color: {text_color};">
                    {token_display}
                    <span class="tooltip">{tooltip_content}</span>
                </span>"""
            
            html_content += """
                </div>
"""
            
            # Statistics
            layer_counts = {}
            for layer in all_layers:
                count = exit_layers.count(layer)
                layer_counts[layer] = count
            
            all_stats[config_name] = layer_counts
            
            stats_text = f"Total tokens: {len(token_strings)} | "
            for layer in all_layers:
                count = layer_counts[layer]
                percentage = (count / len(exit_layers) * 100) if len(exit_layers) > 0 else 0
                layer_name = f"Layer {layer}" if layer != max(all_layers) else "Final"
                stats_text += f"{layer_name}: {count} ({percentage:.1f}%) | "
            
            html_content += f"""
                <div class="stats">{stats_text.rstrip(' |')}</div>
            </div>
"""
        
        # Summary table for this prompt
        html_content += """
            <h3>Summary Statistics</h3>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Configuration</th>
                        <th>Total Tokens</th>
"""
        
        for layer in all_layers:
            layer_name = f"Layer {layer}" if layer != max(all_layers) else "Final"
            html_content += f"""                        <th>{layer_name}</th>
"""
        
        html_content += """                    </tr>
                </thead>
                <tbody>
"""
        
        for config_name in all_stats:
            counts = all_stats[config_name]
            total = sum(counts.values())
            html_content += f"""                    <tr>
                        <td>{config_name}</td>
                        <td>{total}</td>
"""
            for layer in all_layers:
                count = counts.get(layer, 0)
                percentage = (count / total * 100) if total > 0 else 0
                html_content += f"""                        <td>{count} ({percentage:.1f}%)</td>
"""
            html_content += """                    </tr>
"""
        
        html_content += """                </tbody>
            </table>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to: {output_path}")

def visualize_tokens_by_exit_layer(token_strings, exit_layers, early_exit_layer_idxs=None, 
                                  title="Token Early Exit Visualization", save_html=None):
    """
    Visualize tokens colored by their early exit layers in Jupyter notebook or save as HTML.
    
    Args:
        token_strings: List of token strings
        exit_layers: List of exit layer indices (same length as token_strings)
        early_exit_layer_idxs: List/tensor of available early exit layers (optional)
        title: Title for the visualization
        save_html: Path to save HTML file (optional). If provided, saves to file instead of returning HTML object.
    
    Returns:
        IPython.display.HTML object for rendering in notebook (if save_html is None)
        or None (if save_html is provided)
    """
    
    # Get all unique layers and create color mapping
    unique_layers = sorted(set(exit_layers))
    if early_exit_layer_idxs is not None:
        # Include all possible layers even if not used
        all_layers = list(early_exit_layer_idxs) + [27]  # 27 for final layer
        unique_layers = sorted(set(all_layers))
    
    # Create color mapping using coolwarm colormap (blue to red)
    cmap = plt.colormaps.get_cmap('coolwarm_r')
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_layers)-1)
    
    layer_colors = {}
    for i, layer in enumerate(unique_layers):
        color = cmap(norm(i))
        # Convert to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255)
        )
        layer_colors[layer] = hex_color
    
    # Start building HTML
    html_content = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; padding: 20px; 
                background-color: #f9f9f9; border-radius: 10px;">
        <h3 style="text-align: center; color: #333; margin-bottom: 20px;">{title}</h3>
        
        <!-- Legend -->
        <div style="display: flex; justify-content: center; gap: 15px; 
                    margin: 20px 0; padding: 15px; background-color: #fff; 
                    border-radius: 5px; flex-wrap: wrap; border: 1px solid #ddd;">
    """
    
    # Add legend items
    for layer in unique_layers:
        if layer in [l for l in exit_layers]:  # Only show layers that are actually used
            layer_name = f"Layer {layer}" if layer != 27 else "Final Layer"
            color = layer_colors[layer]
            # Determine text color based on background brightness
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "white" if brightness < 128 else "black"
            
            html_content += f"""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 25px; height: 15px; background-color: {color}; 
                                border: 1px solid #333; border-radius: 3px;"></div>
                    <span style="font-size: 14px;">{layer_name}</span>
                </div>
            """
    
    html_content += """
        </div>
        
        <!-- Tokens -->
        <div style="line-height: 2.5; word-wrap: break-word; padding: 15px; 
                    background-color: #fff; border-radius: 5px; border: 1px solid #ddd;">
    """
    
    # Add tokens
    for token, exit_layer in zip(token_strings, exit_layers):
        color = layer_colors[exit_layer]
    # Add tokens
    for token, exit_layer in zip(token_strings, exit_layers):
        color = layer_colors[exit_layer]
        # Escape special characters and handle unicode properly
        token_display = html.escape(token, quote=False)
        # Replace common whitespace and control characters with visible representations
        token_display = token_display.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        # Handle other special characters
        token_display = token_display.replace('\u00a0', '[NBSP]')  # Non-breaking space
        token_display = token_display.replace('\ufeff', '[BOM]')   # Byte order mark
        # Replace any remaining non-printable characters
        token_display = ''.join(char if char.isprintable() or char in ' \n\t' else f'[U+{ord(char):04X}]' for char in token_display)
        
        # Determine text color based on background brightness
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "white" if brightness < 128 else "black"
        
        html_content += f"""<span style="display: inline-block; padding: 4px 8px; margin: 2px; 
                                      border-radius: 4px; border: 1px solid #666; 
                                      font-family: monospace; font-size: 13px; 
                                      background-color: {color}; color: {text_color}; 
                                      font-weight: bold; max-width: 200px; 
                                      overflow-wrap: break-word; vertical-align: middle;">{token_display}</span>"""
    
    html_content += """
        </div>
        
        <!-- Statistics -->
        <div style="margin-top: 15px; padding: 10px; background-color: #e8f4fd; 
                    border-radius: 5px; font-family: monospace; font-size: 13px;">
    """
    
    # Add statistics
    layer_counts = {}
    for layer in unique_layers:
        count = exit_layers.count(layer)
        if count > 0:  # Only show layers that are used
            layer_counts[layer] = count
    
    stats_text = f"Total tokens: {len(token_strings)} | "
    for layer, count in layer_counts.items():
        percentage = (count / len(exit_layers) * 100) if len(exit_layers) > 0 else 0
        layer_name = f"Layer {layer}" if layer != 27 else "Final"
        stats_text += f"{layer_name}: {count} ({percentage:.1f}%) | "
    
    html_content += stats_text.rstrip(' |')
    
    html_content += """
        </div>
    </div>
    """
    
    if save_html:
        # Create complete HTML document for standalone file
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
{html_content}
</body>
</html>"""
        
        # Save to file
        with open(save_html, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"HTML visualization saved to: {save_html}")
        return HTML(html_content)
    else:
        return HTML(html_content)
    
def safe_decode_tokens(tokenizer, token_ids):
    """
    Safely decode tokens: only keep alphanumeric and punctuation characters.
    Everything else shows token ID.
    """
    tokens = []
    for tid in token_ids:
        # Try to decode the token
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        
        # Check if all characters are alphanumeric, space, or punctuation
        if tok and all(c.isalnum() or c.isspace() or c in string.punctuation for c in tok):
            tokens.append(tok)
        else:
            tokens.append(f"[{tid}]")
                
    
    return tokens



def save_results_json(all_results: List[Dict[str, Any]], filename: str = None) -> str:
    """
    Save all_results to a JSON file for later reuse.
    
    Args:
        all_results: The results data structure from your analysis
        filename: Optional filename. If None, uses default based on HTML output
    
    Returns:
        str: Path to the saved JSON file
    """
    if filename is None:
        filename = "tests/early_exit_teacher/data/early_exit_results.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_data = make_serializable(all_results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")
    return filename


def load_results_json(filename: str) -> List[Dict[str, Any]]:
    """
    Load all_results from a JSON file.
    
    Args:
        filename: Path to the JSON file. If None, uses default location
    
    Returns:
        List[Dict]: The loaded all_results data structure
    """
    if not os.path.exists(filename) or filename is None:
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print(f"✅ Loaded results from: {filename}")
    print(f"  - Found {len(all_results)} prompts")
    total_generations = sum(len(p['results']) for p in all_results)
    print(f"  - Total generations: {total_generations}")
    
    return all_results
    """
    Load results from a JSON file.
    
    Args:
        filename: Path to the JSON file
    
    Returns:
        List[Dict]: The loaded results data
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded results from: {filename}")
    return data



def save_multi_prompt_results_html(all_results_data: List[Dict], filename: str = "early_exit_multi_prompt_results.html"):
    """
    Save multi-prompt results and generate single HTML visualization file.
    Also saves the raw data for later reloading.
    """
    html_content = generate_multi_prompt_html_visualization(all_results_data)
    
    # Save HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML visualization saved to: {filename}")
    return filename
import html
import json
from datetime import datetime
from typing import List, Dict, Any

def generate_multi_prompt_html_visualization(all_results_data: List[Dict[str, Any]]) -> str:
    """
    Generate an HTML visualization with hierarchical dashboard structure.
    
    Structure:
    1. Main dashboard with average summary statistics
    2. Individual prompt dashboards with detailed statistics
    3. Expandable mode sections with generation/evaluation tabs
    """
    
    # Map prompts to friendly names based on content
    def get_prompt_name(prompt: str, idx: int) -> str:
        if 'recursion' in prompt.lower():
            return 'Recursion'
        elif 'supervised' in prompt.lower() or 'unsupervised' in prompt.lower():
            return 'Training'
        elif 'http' in prompt.lower():
            return 'HTTP'
        else:
            return f'Prompt {idx + 1}'
    
    # Calculate average statistics across all prompts
    def calculate_average_stats(all_results_data):
        mode_stats = {}
        
        for prompt_data in all_results_data:
            for result in prompt_data['results']:
                mode = result['mode']
                kl_factor = result.get('kl_factor')
                mode_key = f"{mode}_{kl_factor}" if kl_factor is not None else mode
                
                if mode_key not in mode_stats:
                    mode_stats[mode_key] = {
                        'mode': mode,
                        'kl_factor': kl_factor,
                        'count': 0,
                        'coherence_sum': 0,
                        'completeness_sum': 0,
                        'clarity_sum': 0,
                        'no_repetition_sum': 0,
                        'overall_sum': 0,
                        'accuracy_sum': 0,
                        'exit_rate_sum': 0,
                        'total_tokens_sum': 0,
                        'early_exits_sum': 0
                    }
                
                eval_data = result.get('evaluation', {})
                if eval_data and not eval_data.get('error'):
                    stats = mode_stats[mode_key]
                    stats['count'] += 1
                    
                    if eval_data.get('scores'):
                        scores = eval_data['scores']
                        stats['coherence_sum'] += scores.get('coherence', 0)
                        stats['completeness_sum'] += scores.get('completeness', 0)
                        stats['clarity_sum'] += scores.get('clarity', 0)
                        stats['no_repetition_sum'] += scores.get('no_repetition', 0)
                        stats['overall_sum'] += scores.get('overall', 0)
                    
                    if 'accuracy_score' in eval_data:
                        stats['accuracy_sum'] += eval_data['accuracy_score']
                    
                    if eval_data.get('early_exit_stats'):
                        exit_stats = eval_data['early_exit_stats']
                        stats['exit_rate_sum'] += exit_stats.get('early_exit_rate', 0)
                        stats['total_tokens_sum'] += exit_stats.get('total_tokens', 0)
                        stats['early_exits_sum'] += exit_stats.get('early_exits', 0)
        
        # Calculate averages
        avg_stats = []
        for mode_key, stats in mode_stats.items():
            if stats['count'] > 0:
                avg_stat = {
                    'mode': stats['mode'],
                    'kl_factor': stats['kl_factor'],
                    'coherence': round(stats['coherence_sum'] / stats['count'], 1),
                    'completeness': round(stats['completeness_sum'] / stats['count'], 1),
                    'clarity': round(stats['clarity_sum'] / stats['count'], 1),
                    'no_repetition': round(stats['no_repetition_sum'] / stats['count'], 1),
                    'overall': round(stats['overall_sum'] / stats['count'], 1),
                    'accuracy': round(stats['accuracy_sum'] / stats['count'], 3),
                    'exit_rate': round(stats['exit_rate_sum'] / stats['count'], 3),
                    'avg_tokens': round(stats['total_tokens_sum'] / stats['count'], 0),
                    'avg_exits': round(stats['early_exits_sum'] / stats['count'], 0)
                }
                avg_stats.append(avg_stat)
        
        return avg_stats
    
    avg_stats = calculate_average_stats(all_results_data)
    
    # Start building HTML
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Early Exit Generation Analysis - Multi-Level Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: none;
        }
        
        .dashboard.active {
            display: block;
            animation: fadeIn 0.4s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            position: relative;
        }
        
        .dashboard-title {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .dashboard-subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .back-button {
            position: absolute;
            top: 30px;
            right: 40px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid white;
            padding: 10px 25px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .back-button:hover {
            background: white;
            color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .dashboard-content {
            padding: 40px;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }
        
        .stats-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
            position: relative;
            transition: background 0.3s ease;
        }
        
        .stats-table th:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4198 100%);
        }
        
        .stats-table th.sort-asc::after,
        .stats-table th.sort-desc::after {
            content: '';
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
        }
        
        .stats-table th.sort-asc::after {
            border-bottom: 5px solid white;
        }
        
        .stats-table th.sort-desc::after {
            border-top: 5px solid white;
        }
        
        .stats-table td {
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }
        
        .stats-table tr:last-child td {
            border-bottom: none;
        }
        
        .stats-table tr:hover {
            background: #f8f9fa;
        }
        
        .prompt-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 20px;
            background: #f0f4ff;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .prompt-link:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }
        
        .mode-row {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-row.expandable {
            background: #f8f9fa;
        }
        
        .mode-row.expandable:hover {
            background: #e8ecff;
        }
        
        .mode-details {
            display: none;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            margin: 20px 0;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        .mode-details.expanded {
            display: block;
            animation: slideDown 0.3s ease-out;
        }
        
        @keyframes slideDown {
            from { 
                opacity: 0;
                max-height: 0;
                transform: translateY(-10px);
            }
            to { 
                opacity: 1;
                max-height: 1000px;
                transform: translateY(0);
            }
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab-btn {
            flex: 1;
            padding: 15px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            color: #666;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .tab-btn:hover {
            color: #667eea;
            background: white;
        }
        
        .tab-btn.active {
            color: #667eea;
            background: white;
        }
        
        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 3px;
            background: #667eea;
        }
        
        .tab-content {
            display: none;
            padding: 25px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .response-box {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .eval-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .explanation-box {
            background: #fffbf0;
            border: 1px solid #ffd88d;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .explanation-title {
            font-weight: 600;
            color: #d4841f;
            margin-bottom: 10px;
        }
        
        .prompt-info {
            background: #f0f4ff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .prompt-text {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }
        
        .expand-icon {
            display: inline-block;
            margin-right: 10px;
            transition: transform 0.3s ease;
        }
        
        .mode-row.expanded .expand-icon {
            transform: rotate(90deg);
        }
        
        .mode-display {
            font-weight: 600;
            color: #333;
        }
        
        .kl-badge {
            background: #e8ecff;
            color: #667eea;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 8px;
        }
    </style>
</head>
<body>
"""
    
    # Main Dashboard
    html_content += """
    <div class="dashboard active" id="main-dashboard">
        <div class="dashboard-header">
            <h1 class="dashboard-title">📊 Early Exit Generation Analysis</h1>
            <p class="dashboard-subtitle">Average Performance Across All Prompts</p>
        </div>
        <div class="dashboard-content">
            <table class="stats-table" id="main-stats-table">
                <thead>
                    <tr>
                        <th onclick="sortTable('main-stats-table', 0)">Configuration</th>
                        <th onclick="sortTable('main-stats-table', 1)">Coherence</th>
                        <th onclick="sortTable('main-stats-table', 2)">Completeness</th>
                        <th onclick="sortTable('main-stats-table', 3)">Clarity</th>
                        <th onclick="sortTable('main-stats-table', 4)">No Repetition</th>
                        <th onclick="sortTable('main-stats-table', 5)">Overall</th>
                        <th onclick="sortTable('main-stats-table', 6)">Overall (%)</th>
                        <th onclick="sortTable('main-stats-table', 7)">Avg Tokens</th>
                        <th onclick="sortTable('main-stats-table', 8)">Avg Exits</th>
                        <th onclick="sortTable('main-stats-table', 9)">Exit Rate</th>
                        <th>Promptwise Details</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rows for average statistics
    for stat in avg_stats:
        mode_display = stat['mode'].replace('_', ' ').title()
        if stat['kl_factor'] is not None:
            mode_display += f' <span class="kl-badge">KL: {stat["kl_factor"]}</span>'
        
        # Create links to individual prompts
        prompt_links = []
        for idx, prompt_data in enumerate(all_results_data):
            prompt_name = get_prompt_name(prompt_data['prompt'], idx)
            prompt_links.append(f'<a href="#" onclick="showPromptDashboard({idx}); return false;" class="prompt-link">{prompt_name}</a>')
        
        html_content += f"""
                    <tr>
                        <td><span class="mode-display">{mode_display}</span></td>
                        <td>{stat['coherence']}/10</td>
                        <td>{stat['completeness']}/10</td>
                        <td>{stat['clarity']}/10</td>
                        <td>{stat['no_repetition']}/10</td>
                        <td><strong>{stat['overall']}/40</strong></td>
                        <td>{stat['accuracy']*100:.1f}%</td>
                        <td>{stat['avg_tokens']:.0f}</td>
                        <td>{stat['avg_exits']:.0f}</td>
                        <td>{stat['exit_rate']*100:.1f}%</td>
                        <td>{' '.join(prompt_links)}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
    
    # Individual Prompt Dashboards
    for prompt_idx, prompt_data in enumerate(all_results_data):
        prompt = prompt_data['prompt']
        prompt_name = get_prompt_name(prompt, prompt_idx)
        results = prompt_data['results']
        
        html_content += f"""
    <div class="dashboard" id="prompt-dashboard-{prompt_idx}">
        <div class="dashboard-header">
            <h1 class="dashboard-title">📝 {html.escape(prompt_name)} Analysis</h1>
            <p class="dashboard-subtitle">Detailed Performance Metrics</p>
            <button class="back-button" onclick="showMainDashboard()">← Back to Overview</button>
        </div>
        <div class="dashboard-content">
            <div class="prompt-info">
                <h3 style="margin-bottom: 10px; color: #667eea;">Prompt:</h3>
                <div class="prompt-text">{html.escape(prompt)}</div>
            </div>
            
            <table class="stats-table" id="prompt-table-{prompt_idx}">
                <thead>
                    <tr>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 0)">Configuration</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 1)">Coherence</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 2)">Completeness</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 3)">Clarity</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 4)">No Repetition</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 5)">Overall</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 6)">Overall (%)</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 7)">Total Tokens</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 8)">Early Exits</th>
                        <th onclick="sortTable('prompt-table-{prompt_idx}', 9)">Exit Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add rows for each mode with expandable details
        for result_idx, result in enumerate(results):
            mode = result['mode']
            kl_factor = result.get('kl_factor')
            response = result['response']
            evaluation = result.get('evaluation', {})
            
            mode_display = mode.replace('_', ' ').title()
            if kl_factor is not None:
                mode_display += f' <span class="kl-badge">KL: {kl_factor}</span>'
            
            # Extract metrics
            metrics = {'coherence': 'N/A', 'completeness': 'N/A', 'clarity': 'N/A', 
                      'no_repetition': 'N/A', 'overall': 'N/A', 'accuracy': 0,
                      'total_tokens': 'N/A', 'early_exits': 'N/A', 'exit_rate': 0}
            
            if evaluation and not evaluation.get('error'):
                if evaluation.get('scores'):
                    scores = evaluation['scores']
                    metrics.update({k: scores.get(k, 'N/A') for k in ['coherence', 'completeness', 'clarity', 'no_repetition', 'overall']})
                if 'accuracy_score' in evaluation:
                    metrics['accuracy'] = evaluation['accuracy_score']
                if evaluation.get('early_exit_stats'):
                    exit_stats = evaluation['early_exit_stats']
                    metrics['total_tokens'] = exit_stats.get('total_tokens', 'N/A')
                    metrics['early_exits'] = exit_stats.get('early_exits', 'N/A')
                    metrics['exit_rate'] = exit_stats.get('early_exit_rate', 0)
            
            html_content += f"""
                    <tr class="mode-row expandable" onclick="toggleModeDetails({prompt_idx}, {result_idx})">
                        <td>
                            <span class="expand-icon">▶</span>
                            <span class="mode-display">{mode_display}</span>
                        </td>
                        <td>{metrics['coherence']}</td>
                        <td>{metrics['completeness']}</td>
                        <td>{metrics['clarity']}</td>
                        <td>{metrics['no_repetition']}</td>
                        <td><strong>{metrics['overall']}</strong></td>
                        <td>{metrics['accuracy']*100:.1f}%</td>
                        <td>{metrics['total_tokens']}</td>
                        <td>{metrics['early_exits']}</td>
                        <td>{metrics['exit_rate']*100:.1f}%</td>
                    </tr>
                    <tr>
                        <td colspan="10" style="padding: 0; border: none;">
                            <div class="mode-details" id="mode-details-{prompt_idx}-{result_idx}">
                                <div class="tabs">
                                    <button class="tab-btn active" onclick="showTab({prompt_idx}, {result_idx}, 'generation')">
                                        📝 Generation
                                    </button>
                                    <button class="tab-btn" onclick="showTab({prompt_idx}, {result_idx}, 'evaluation')">
                                        📊 Evaluation
                                    </button>
                                </div>
                                
                                <div class="tab-content active" id="tab-{prompt_idx}-{result_idx}-generation">
                                    <div class="response-box">{html.escape(response)}</div>
                                </div>
                                
                                <div class="tab-content" id="tab-{prompt_idx}-{result_idx}-evaluation">
"""
            
            if evaluation and not evaluation.get('error'):
                html_content += """                                    <div class="eval-metrics">
"""
                
                # Add metric cards
                if evaluation.get('scores'):
                    for key, label in [('coherence', 'Coherence'), ('completeness', 'Completeness'), 
                                      ('clarity', 'Clarity'), ('no_repetition', 'No Repetition')]:
                        if key in evaluation['scores']:
                            html_content += f"""                                        <div class="metric-card">
                                            <div class="metric-value">{evaluation['scores'][key]}/10</div>
                                            <div class="metric-label">{label}</div>
                                        </div>
"""
                
                if 'accuracy_score' in evaluation:
                    html_content += f"""                                        <div class="metric-card">
                                            <div class="metric-value">{evaluation['accuracy_score']*100:.1f}%</div>
                                            <div class="metric-label">Overall (%)</div>
                                        </div>
"""
                
                html_content += """                                    </div>
"""
                
                # Add brief explanation if available
                if evaluation.get('evaluation_text'):
                    # Extract the brief explanation from evaluation_text
                    eval_text = evaluation['evaluation_text']
                    if 'Brief explanation:' in eval_text:
                        brief_exp = eval_text.split('Brief explanation:')[1].strip()
                        html_content += f"""                                    <div class="explanation-box">
                                        <div class="explanation-title">💡 Brief Explanation</div>
                                        <div>{html.escape(brief_exp)}</div>
                                    </div>
"""
                elif evaluation.get('brief_explanation'):
                    html_content += f"""                                    <div class="explanation-box">
                                        <div class="explanation-title">💡 Brief Explanation</div>
                                        <div>{html.escape(evaluation['brief_explanation'])}</div>
                                    </div>
"""
                elif evaluation.get('explanation'):
                    html_content += f"""                                    <div class="explanation-box">
                                        <div class="explanation-title">💡 Explanation</div>
                                        <div>{html.escape(evaluation['explanation'])}</div>
                                    </div>
"""
            else:
                error_msg = evaluation.get('error', 'No evaluation data available') if evaluation else 'No evaluation data available'
                html_content += f"""                                    <div class="explanation-box">
                                        <div class="explanation-title">⚠️ Error</div>
                                        <div>{html.escape(error_msg)}</div>
                                    </div>
"""
            
            html_content += """                                </div>
                            </div>
                        </td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
    
    # Add JavaScript
    html_content += """
    <script>
        // Sorting functionality
        const sortStates = {};
        
        function sortTable(tableId, columnIndex) {
            const table = document.getElementById(tableId);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr')).filter(row => !row.querySelector('.mode-details'));
            const th = table.querySelectorAll('th')[columnIndex];
            
            // Initialize sort state for this table if not exists
            if (!sortStates[tableId]) {
                sortStates[tableId] = {};
            }
            
            // Get current sort state
            const currentSort = sortStates[tableId][columnIndex] || 'none';
            let newSort = currentSort === 'none' ? 'asc' : currentSort === 'asc' ? 'desc' : 'asc';
            
            // Reset all other columns' visual state
            table.querySelectorAll('th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Set visual state for current column
            th.classList.add('sort-' + newSort);
            
            // Store sort state
            sortStates[tableId][columnIndex] = newSort;
            
            // Perform sort
            const sortedRows = rows.sort((a, b) => {
                let aValue = a.cells[columnIndex].textContent.trim();
                let bValue = b.cells[columnIndex].textContent.trim();
                
                // Remove HTML tags and get text content
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = aValue;
                aValue = tempDiv.textContent || tempDiv.innerText || '';
                tempDiv.innerHTML = bValue;
                bValue = tempDiv.textContent || tempDiv.innerText || '';
                
                // Extract numeric values if present
                const aNum = parseFloat(aValue.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bValue.replace(/[^0-9.-]/g, ''));
                
                let comparison = 0;
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    comparison = aNum - bNum;
                } else {
                    comparison = aValue.localeCompare(bValue);
                }
                
                return newSort === 'asc' ? comparison : -comparison;
            });
            
            // Clear tbody and add sorted rows
            tbody.innerHTML = '';
            sortedRows.forEach(row => {
                tbody.appendChild(row);
                // Also append the details row if it exists
                const nextRow = row.nextElementSibling;
                if (nextRow && nextRow.querySelector('.mode-details')) {
                    tbody.appendChild(nextRow);
                }
            });
        }
        
        function showMainDashboard() {
            document.querySelectorAll('.dashboard').forEach(d => d.classList.remove('active'));
            document.getElementById('main-dashboard').classList.add('active');
        }
        
        function showPromptDashboard(idx) {
            document.querySelectorAll('.dashboard').forEach(d => d.classList.remove('active'));
            document.getElementById('prompt-dashboard-' + idx).classList.add('active');
        }
        
        function toggleModeDetails(promptIdx, resultIdx) {
            const detailsId = 'mode-details-' + promptIdx + '-' + resultIdx;
            const details = document.getElementById(detailsId);
            const row = details.closest('tr').previousElementSibling;
            
            if (details.classList.contains('expanded')) {
                details.classList.remove('expanded');
                row.classList.remove('expanded');
            } else {
                // Close other expanded details in the same table
                const table = details.closest('table');
                table.querySelectorAll('.mode-details.expanded').forEach(d => {
                    d.classList.remove('expanded');
                    d.closest('tr').previousElementSibling.classList.remove('expanded');
                });
                
                details.classList.add('expanded');
                row.classList.add('expanded');
            }
        }
        
        function showTab(promptIdx, resultIdx, tab) {
            const prefix = 'tab-' + promptIdx + '-' + resultIdx + '-';
            
            // Hide all tabs
            document.getElementById(prefix + 'generation').classList.remove('active');
            document.getElementById(prefix + 'evaluation').classList.remove('active');
            
            // Show selected tab
            document.getElementById(prefix + tab).classList.add('active');
            
            // Update tab buttons
            const details = document.getElementById('mode-details-' + promptIdx + '-' + resultIdx);
            details.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().includes(tab)) {
                    btn.classList.add('active');
                }
            });
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const activePromptDashboard = document.querySelector('.dashboard:not(#main-dashboard).active');
                if (activePromptDashboard) {
                    showMainDashboard();
                }
            }
        });
    </script>
</body>
</html>
"""
    
    return html_content