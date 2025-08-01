import os
import html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_html_visualization(all_results, early_exit_layer_idxs, test_prompts,
                             output_path='tests/prompt_based_kl_output.html'):
    """
    Create an HTML file with visualization of early exit behavior across different KL strengths.
    
    Args:
        all_results: Dictionary with structure {kl_strength: {sentence_idx: (tokens, exit_layers, text)}}
        early_exit_layer_idxs: Tensor of available early exit layers
        test_prompts: List of test prompts
        output_path: Path to save the HTML file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create color mapping
    all_layers = list(early_exit_layer_idxs.numpy()) + [-1]  # -1 represents final layer
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
    
    # Start building HTML
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Early Exit Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        h2 {
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 {
            color: #666;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-box {
            width: 30px;
            height: 20px;
            border: 1px solid #333;
            border-radius: 3px;
        }
        .tokens-container {
            margin: 15px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
            line-height: 2.2;
            word-wrap: break-word;
        }
        .token {
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
        }
        .stats {
            margin: 15px 0;
            padding: 10px;
            background-color: #e8f4fd;
            border-radius: 5px;
            font-family: monospace;
            font-size: 13px;
        }
        .summary-table {
            margin: 20px 0;
            border-collapse: collapse;
            width: 100%;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .summary-table th {
            background-color: #4CAF50;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .prompt-section {
            margin-bottom: 50px;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background-color: #fdfdfd;
        }
        .no-data {
            font-style: italic;
            color: #999;
            padding: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Early Exit Behavior Visualization</h1>
        
        <!-- Color Legend -->
        <div class="legend">
"""
    
    # Add legend items
    for layer in all_layers:
        layer_name = f"Layer {layer}" if layer != -1 else "Final Layer"
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
            <h3>KL Strength: {kl_strength}</h3>
"""
            
            if prompt_idx not in all_results[kl_strength]:
                html_content += """
            <div class="no-data">No data available</div>
"""
                continue
            
            token_strings, exit_layers, _ = all_results[kl_strength][prompt_idx]
            
            # Display tokens
            html_content += """
            <div class="tokens-container">
"""
            
            for token, exit_layer in zip(token_strings, exit_layers):
                color = layer_colors[exit_layer]
                # Escape special characters in token
                token_display = html.escape(token).replace('\n', '\\n').replace('\t', '\\t')
                
                # Determine text color based on background
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = "white" if brightness < 128 else "black"
                
                html_content += f"""<span class="token" style="background-color: {color}; color: {text_color};">{token_display}</span>"""
            
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
                layer_name = f"Layer {layer}" if layer != -1 else "Final"
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
                        <th>KL Strength</th>
                        <th>Total Tokens</th>
"""
        
        for layer in all_layers:
            layer_name = f"Layer {layer}" if layer != -1 else "Final"
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