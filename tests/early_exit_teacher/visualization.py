import os
import html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display


import os
import html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display


def create_html_visualization(all_results, early_exit_layer_idxs, test_prompts,
                             output_path='tests/prompt_based_kl_output.html',
                             title='Early Exit Behavior Visualization'):
    """
    Create an HTML file with visualization of early exit behavior across different KL strengths.
    
    Args:
        all_results: Dictionary with structure {kl_strength: {sentence_idx: (tokens, exit_layers, text, kl_divs)}}
                    Note: kl_divs is now expected as the 4th element of the tuple
        early_exit_layer_idxs: Tensor of available early exit layers
        test_prompts: List of test prompts
        output_path: Path to save the HTML file
        title: Custom title for the visualization (default: 'Early Exit Behavior Visualization')
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create color mapping
    all_layers = list(early_exit_layer_idxs.numpy()) + [27]  # 27 represents final layer
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
    
    kl_strengths = sorted(all_results.keys())
    
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
        layer_name = f"Layer {layer}" if layer != 27 else "Final Layer"
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
        
        # Process each KL strength for this prompt
        all_stats = {}
        for kl_strength in kl_strengths:
            html_content += f"""
            <h3>{kl_strength}</h3>
"""
            
            if prompt_idx not in all_results[kl_strength]:
                html_content += """
            <div class="no-data">No data available</div>
"""
                continue
            
            result_tuple = all_results[kl_strength][prompt_idx]
            if len(result_tuple) == 4:
                token_strings, exit_layers, _, kl_divs = result_tuple
            else:
                # Backward compatibility if kl_divs not provided
                token_strings, exit_layers, _ = result_tuple
                kl_divs = [None] * len(token_strings)
            
            # Display tokens
            html_content += """
            <div class="tokens-container">
"""
            
            for i, (token, exit_layer) in enumerate(zip(token_strings, exit_layers)):
                color = layer_colors[exit_layer]
                # Escape special characters in token
                token_display = html.escape(token).replace('\n', '\\n').replace('\t', '\\t')
                
                # Determine text color based on background
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = "white" if brightness < 128 else "black"
                
                # Create tooltip content
                layer_display = "Final Layer" if exit_layer == 27 else f"Layer {exit_layer}"
                tooltip_content = f"Exit Layer: {layer_display}"
                
                if i < len(kl_divs) and kl_divs[i] is not None:
                    # Handle 1D tensor case - kl_divs[i] is a tensor of shape [num_layers]
                    if hasattr(kl_divs[i], 'shape') and len(kl_divs[i].shape) == 1:
                        # kl_divs[i] is already a 1D tensor with KL values for each layer
                        tooltip_lines = [f"Exit Layer: {layer_display}"]
                        tooltip_lines.append("KL Divergences:")
                        for j, kl_val in enumerate(kl_divs[i]):
                            if j < len(early_exit_layer_idxs):
                                layer_idx = early_exit_layer_idxs[j].item()
                                tooltip_lines.append(f"Layer {layer_idx}: {kl_val.item():.2f}")
                        tooltip_content = "<br>".join(tooltip_lines)
                    elif hasattr(kl_divs[i], 'item'):
                        # Single scalar tensor
                        tooltip_content += f"<br>KL Divergence: {kl_divs[i].item():.2f}"
                    else:
                        # Fallback for other cases
                        tooltip_content += f"<br>KL Divergence: {float(kl_divs[i]):.2f}"
                
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
            
            all_stats[kl_strength] = layer_counts
            
            stats_text = f"Total tokens: {len(token_strings)} | "
            for layer in all_layers:
                count = layer_counts[layer]
                percentage = (count / len(exit_layers) * 100) if len(exit_layers) > 0 else 0
                layer_name = f"Layer {layer}" if layer != 27 else "Final"
                stats_text += f"{layer_name}: {count} ({percentage:.1f}%) | "
            
            html_content += f"""
            <div class="stats">{stats_text.rstrip(' |')}</div>
"""
        
        # Summary table for this prompt
        html_content += """
            <h3>Summary Statistics</h3>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Mode</th>
                        <th>Total Tokens</th>
"""
        
        for layer in all_layers:
            layer_name = f"Layer {layer}" if layer != 27 else "Final"
            html_content += f"""                        <th>{layer_name}</th>
"""
        
        html_content += """                    </tr>
                </thead>
                <tbody>
"""
        
        for kl_strength in kl_strengths:
            if kl_strength in all_stats:
                counts = all_stats[kl_strength]
                total = sum(counts.values())
                html_content += f"""                    <tr>
                        <td>{kl_strength}</td>
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
        # Escape special characters
        token_display = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        token_display = token_display.replace('\n', '\\n').replace('\t', '\\t')
        
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