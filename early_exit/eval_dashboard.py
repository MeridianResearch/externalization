"""
Model Evaluation Dashboard Generator
Creates an interactive HTML dashboard from evaluation CSV files
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
import argparse


class EvaluationDashboard:
    """Main class for creating evaluation dashboard from CSV files"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.csv_files = list(self.folder_path.glob("rollout_results_*.csv"))
        self.evaluations = []
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse hyperparameters from filename
        Pattern: rollout_results_{model_path}_{mode}_{n}_{k}_{config_mode}.csv
        """
        # Remove extension and prefix
        name = Path(filename).stem
        if name.startswith("rollout_results_"):
            name = name[len("rollout_results_"):]
        
        # Split by underscores
        parts = name.split("_")
        
        hyperparams = {}
        
        # Work backwards from the end to handle model_path with underscores
        if len(parts) >= 4:
            # Last part should be config_mode (like 'greedy')
            if not parts[-1].isdigit():
                hyperparams['sampling'] = parts[-1]
                if hyperparams['sampling'] == 'deepseek':
                    hyperparams['sampling'] = 'T=1'
                parts = parts[:-1]
            else:
                hyperparams['sampling'] = 'T=1'
            
            # Next should be k (number)
            if parts and parts[-1].isdigit():
                hyperparams['k'] = parts[-1]
                parts = parts[:-1]
            
            # Next should be n (number)
            if parts and parts[-1].isdigit():
                hyperparams['n'] = int(parts[-1])  # Convert to int for filtering
                parts = parts[:-1]
            
            # Next should be mode - handle both 'off' and 'free_generate'
            if parts:
                # Check for 'free_generate' (two parts)
                if len(parts) >= 2 and parts[-2] == 'free' and parts[-1] == 'generate':
                    hyperparams['mode'] = 'free_generate'
                    parts = parts[:-2]
                # Check for 'off' (one part)
                elif parts[-1] == 'off':
                    hyperparams['mode'] = 'off'
                    parts = parts[:-1]
                else:
                    # If no recognized mode, assume it's part of model_path
                    hyperparams['mode'] = 'unknown'
            
            # Remaining parts form the model_path
            if parts:
                hyperparams['model_path'] = "_".join(parts)
        
        return hyperparams
    
    def parse_completion_text(self, completion_text: str) -> Tuple[str, str]:
        """
        Parse completion text to extract Chain of Thought and Response
        Returns (chain_of_thought, response)
        """
        if pd.isna(completion_text) or not completion_text:
            return "", ""
        
        completion_text = str(completion_text)
        
        # Remove EOS token if present
        completion_text = completion_text.replace('<|endoftext|>', '').strip()
        
        # Pattern to match <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, completion_text, re.DOTALL)
        
        # Extract chain of thought (content within think tags)
        chain_of_thought = ""
        if think_matches:
            chain_of_thought = "\n".join(think_matches).strip()
        
        # Extract response (everything after the last </think> tag)
        response = completion_text
        if think_matches:
            # Find the last </think> tag and take everything after it
            last_think_end = completion_text.rfind('</think>')
            if last_think_end != -1:
                response = completion_text[last_think_end + len('</think>'):].strip()
        
        return chain_of_thought, response
    
    def process_completion_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataframe to add Chain of Thought and Response columns
        """
        df = df.copy()
        
        # Check if completion_text column exists
        completion_col = None
        for col in df.columns:
            if 'completion_text' in col.lower():
                completion_col = col
                break
        
        if completion_col is None:
            # If no completion_text column found, add empty columns
            df['samples/chain_of_thought'] = ""
            df['samples/response'] = ""
            return df
        
        # Parse completion text for each row
        cot_data = []
        response_data = []
        
        for _, row in df.iterrows():
            completion_text = row[completion_col]
            cot, response = self.parse_completion_text(completion_text)
            cot_data.append(cot)
            response_data.append(response)
        
        # Add new columns
        df['samples/chain_of_thought'] = cot_data
        df['samples/response'] = response_data
        
        # Remove the original completion_text column
        df = df.drop(columns=[completion_col])
        prompt_col = self._find_prompt_column(df)
        cot_response_cols = ["samples/chain_of_thought", "samples/response"]
        if prompt_col:
            df = self._move_columns_after(df, cot_response_cols, prompt_col)
        return df
    
    def calculate_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics from dataframe"""
        summary = {}
        
        # Overall accuracy
        if 'samples/verify_reward' in df.columns:
            summary['overall_accuracy'] = df['samples/verify_reward'].mean()
        
        # Total rollouts
        summary['total_rollouts'] = len(df)
        
        # Unique examples
        if 'example' in df.columns:
            summary['unique_examples'] = df['example'].nunique()
        
        # EOS token statistics
        if 'samples/contains_eos' in df.columns:
            eos_count = df['samples/contains_eos'].sum()
            summary['eos_count'] = int(eos_count)
            summary['eos_ratio'] = eos_count / len(df) if len(df) > 0 else 0
        
        # Difficulty breakdown
        if 'samples/difficulty_category' in df.columns and 'samples/verify_reward' in df.columns:
            difficulty_stats = {}
            for difficulty in df['samples/difficulty_category'].unique():
                mask = df['samples/difficulty_category'] == difficulty
                rewards = df.loc[mask, 'samples/verify_reward']
                difficulty_stats[difficulty] = {
                    'accuracy': rewards.mean(),
                    'count': len(rewards)
                }
            summary['difficulty_stats'] = difficulty_stats
        
        return summary
    
    def clean_column_name(self, column_name: str) -> str:
        """Remove 'samples/' prefix from column names for display"""
        if column_name.startswith('samples/'):
            return column_name[8:]  # Remove 'samples/' prefix
        return column_name
    
    def process_csv_files(self):
        """Process all CSV files in the folder"""
        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)
                hyperparams = self.parse_filename(csv_file.name)
                
                # Only include evaluations where n > 10
                if hyperparams.get('n', 0) > 10:
                    # Process completion text to extract CoT and Response
                    df = self.process_completion_columns(df)
                    
                    summary = self.calculate_summary(df)
                    
                    # Store CSV data as JSON for embedding in HTML
                    csv_data = df.to_dict('records')
                    
                    # Create cleaned column names for display
                    cleaned_columns = [self.clean_column_name(col) for col in df.columns]
                    
                    self.evaluations.append({
                        'filename': csv_file.name,
                        'hyperparams': hyperparams,
                        'summary': summary,
                        'data': csv_data,
                        'columns': list(df.columns),  # Original column names for data access
                        'display_columns': cleaned_columns  # Cleaned column names for display
                    })
                else:
                    print(f"Skipping {csv_file.name}: n={hyperparams.get('n', 0)} <= 10")
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
    
    def _find_prompt_column(self, df: pd.DataFrame) -> str | None:
        """
        Try common prompt column names; fall back to any column containing
        'prompt'/'question'/'input' (case-insensitive).
        """
        preferred = [
            "samples/prompt", "samples/prompt_text", "samples/question",
            "prompt", "prompt_text", "question", "input", "samples/input"
        ]
        for name in preferred:
            if name in df.columns:
                return name
        for col in df.columns:
            low = col.lower()
            if "prompt" in low or "question" in low or "input" in low:
                return col
        return None

    def _move_columns_after(self, df: pd.DataFrame, cols_to_move: list[str], after_col: str) -> pd.DataFrame:
        """
        Reorder DataFrame so that cols_to_move appear right after after_col.
        Non-destructive to other columns' relative order.
        """
        cols_to_move = [c for c in cols_to_move if c in df.columns]
        if not cols_to_move or after_col not in df.columns:
            return df
        base = [c for c in df.columns if c not in cols_to_move]
        insert_at = base.index(after_col) + 1
        new_order = base[:insert_at] + cols_to_move + base[insert_at:]
        return df[new_order]
    def generate_html(self) -> str:
        """Generate the complete HTML dashboard"""
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        h1 {
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            font-size: 2em;
        }
        
        /* Main Page Styles */
        .main-page {
            padding: 30px;
        }
        
        .evaluations-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .evaluations-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9em;
            cursor: pointer;
            user-select: none;
            position: relative;
            transition: background-color 0.2s ease;
        }
        
        .evaluations-table th:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
        
        .evaluations-table th .sort-indicator {
            display: inline-block;
            margin-left: 8px;
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .evaluations-table th.sorted-asc .sort-indicator::after {
            content: "▲";
        }
        
        .evaluations-table th.sorted-desc .sort-indicator::after {
            content: "▼";
        }
        
        .evaluations-table th:not(.sorted-asc):not(.sorted-desc) .sort-indicator::after {
            content: "⇅";
        }
        
        .evaluations-table td {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: middle;
        }
        
        .evaluations-table tr {
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        
        .evaluations-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        
        .evaluations-table tr:last-child td {
            border-bottom: none;
        }
        
        .model-name {
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }
        
        .accuracy-badge {
            background: linear-gradient(135deg, #4caf50, #8bc34a);
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
            display: inline-block;
        }
        
        .param-value {
            color: #667eea;
            font-weight: 600;
        }
        
        .mode-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .mode-base {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .mode-sft {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .mode-rl {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .mode-unknown {
            background: #fafafa;
            color: #757575;
        }
        
        .stat-small {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Detail Page Styles */
        .detail-page {
            display: none;
            padding: 20px;
        }
        
        .back-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin-bottom: 20px;
            transition: opacity 0.3s ease;
        }
        
        .back-button:hover {
            opacity: 0.9;
        }
        
        .detail-summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .hyperparams-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .param-item {
            background: white;
            padding: 10px;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .param-label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .stat-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .difficulty-breakdown {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .difficulty-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .difficulty-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .difficulty-item:last-child {
            border-bottom: none;
        }
        
        .accuracy-bar {
            display: inline-block;
            width: 100px;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-left: 10px;
            vertical-align: middle;
        }
        
        .accuracy-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            table-layout: fixed;
        }
        
        .detail-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s ease;
            font-size: 11px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .detail-table th:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
        
        .detail-table th .sort-indicator {
            display: inline-block;
            margin-left: 8px;
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .detail-table th.sorted-asc .sort-indicator::after {
            content: "▲";
        }
        
        .detail-table th.sorted-desc .sort-indicator::after {
            content: "▼";
        }
        
        .detail-table th:not(.sorted-asc):not(.sorted-desc) .sort-indicator::after {
            content: "⇅";
        }
        
        .detail-table td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
        }
        
        .detail-table td:hover {
            background: #f0f0f0;
            max-width: none;
            white-space: normal;
            word-break: break-word;
        }
        
        .detail-table tr:hover {
            background: #f9f9f9;
        }
        
        /* Special styling for Chain of Thought and Response columns */
        .detail-table td.cot-column {
            max-width: 400px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            background-color: #f8f9fa;
        }
        
        .detail-table td.response-column {
            max-width: 400px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            background-color: #fff8e1;
        }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 10px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #667eea;
        }
        
        .cell-content {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Evaluation Dashboard</h1>
        
        <div class="main-page" id="mainPage">
            <table class="evaluations-table" id="evaluationsTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('model')" data-column="model">
                            Model Path<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('mode')" data-column="mode">
                            Type<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('n')" data-column="n">
                            N<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('k')" data-column="k">
                            K<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('sampling')" data-column="sampling">
                            Sampling<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('accuracy')" data-column="accuracy">
                            Accuracy<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('rollouts')" data-column="rollouts">
                            Rollouts<span class="sort-indicator"></span>
                        </th>
                        <th onclick="sortTable('eos_rate')" data-column="eos_rate">
                            EOS Rate<span class="sort-indicator"></span>
                        </th>
                    </tr>
                </thead>
                <tbody id="evaluationsBody">
                </tbody>
            </table>
        </div>
        
        <div class="detail-page" id="detailPage">
            <button class="back-button" onclick="showMainPage()">← Back to Dashboard</button>
            <div class="detail-summary" id="detailSummary"></div>
            <div class="table-container">
                <table class="detail-table" id="detailTable"></table>
            </div>
        </div>
    </div>
    
    <div id="cellModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h3>Cell Content</h3>
            <div id="modalContent" class="cell-content"></div>
        </div>
    </div>
    
    <script>
        // Embed evaluation data
        const evaluations = ''' + json.dumps(self.evaluations) + ''';
        
        let currentEvaluation = null;
        let currentSort = { column: null, direction: 'asc' };
        let currentDetailSort = { column: null, direction: 'asc' };
        let sortedEvaluations = [...evaluations];
        let sortedDetailData = [];
        
        function formatPercent(value) {
            return (value * 100).toFixed(2) + '%';
        }
        
        function getModelType(modelPath, mode) {
            if (mode === 'off') return 'Base';
            if (modelPath && modelPath.includes('gsm8k_old_school_1')) return 'SFT';
            if (modelPath && modelPath.includes('epoch-')) return 'RL';
            return 'Unknown';
        }
        
        function getModeClass(modelPath, mode) {
            const modelType = getModelType(modelPath, mode);
            if (modelType === 'Base') return 'mode-base';
            if (modelType === 'SFT') return 'mode-sft';
            if (modelType === 'RL') return 'mode-rl';
            return 'mode-unknown';
        }
        
        function getValue(eval, column) {
            const hyperparams = eval.hyperparams;
            const summary = eval.summary;
            
            switch(column) {
                case 'model':
                    return hyperparams.model_path || 'Unknown Model';
                case 'mode':
                    return getModelType(hyperparams.model_path, hyperparams.mode);
                case 'n':
                    return parseInt(hyperparams.n) || 0;
                case 'k':
                    return parseInt(hyperparams.k) || 1;
                case 'sampling':
                    return hyperparams.sampling || 'T=1';
                case 'accuracy':
                    return summary.overall_accuracy || 0;
                case 'rollouts':
                    return summary.total_rollouts || 0;
                case 'eos_rate':
                    return summary.eos_ratio || 0;
                default:
                    return '';
            }
        }
        
        function getDetailValue(row, originalColumn) {
            const value = row[originalColumn];
            if (value === null || value === undefined) return '';
            return value;
        }
        
        function sortTable(column) {
            // If same column, toggle direction
            if (currentSort.column === column) {
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            } else {
                currentSort.column = column;
                currentSort.direction = 'asc';
            }
            
            // Sort evaluations
            sortedEvaluations.sort((a, b) => {
                const valueA = getValue(a, column);
                const valueB = getValue(b, column);
                
                let comparison = 0;
                if (typeof valueA === 'number' && typeof valueB === 'number') {
                    comparison = valueA - valueB;
                } else {
                    comparison = String(valueA).localeCompare(String(valueB));
                }
                
                return currentSort.direction === 'asc' ? comparison : -comparison;
            });
            
            // Update header styling
            document.querySelectorAll('.evaluations-table th').forEach(th => {
                th.classList.remove('sorted-asc', 'sorted-desc');
            });
            
            const currentHeader = document.querySelector(`[data-column="${column}"]`);
            currentHeader.classList.add(currentSort.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
            
            // Re-render table
            renderTable();
        }
        
        function sortDetailTable(originalColumn) {
            // If same column, toggle direction
            if (currentDetailSort.column === originalColumn) {
                currentDetailSort.direction = currentDetailSort.direction === 'asc' ? 'desc' : 'asc';
            } else {
                currentDetailSort.column = originalColumn;
                currentDetailSort.direction = 'asc';
            }
            
            // Sort detail data
            sortedDetailData.sort((a, b) => {
                const valueA = getDetailValue(a, originalColumn);
                const valueB = getDetailValue(b, originalColumn);
                
                let comparison = 0;
                
                // Try to parse as numbers first
                const numA = parseFloat(valueA);
                const numB = parseFloat(valueB);
                
                if (!isNaN(numA) && !isNaN(numB)) {
                    comparison = numA - numB;
                } else {
                    comparison = String(valueA).localeCompare(String(valueB));
                }
                
                return currentDetailSort.direction === 'asc' ? comparison : -comparison;
            });
            
            // Update header styling for detail table
            document.querySelectorAll('.detail-table th').forEach(th => {
                th.classList.remove('sorted-asc', 'sorted-desc');
            });
            
            const currentHeader = document.querySelector(`[data-detail-column="${originalColumn}"]`);
            if (currentHeader) {
                currentHeader.classList.add(currentDetailSort.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
            }
            
            // Re-render detail table
            renderDetailTable();
        }
        
        function createEvaluationRow(eval, index) {
            const accuracy = eval.summary.overall_accuracy || 0;
            const hyperparams = eval.hyperparams;
            const modelType = getModelType(hyperparams.model_path, hyperparams.mode);
            
            return `
                <tr onclick="showDetail(${index})">
                    <td>
                        <div class="model-name">${hyperparams.model_path || 'Unknown Model'}</div>
                    </td>
                    <td>
                        <span class="mode-badge ${getModeClass(hyperparams.model_path, hyperparams.mode)}">${modelType}</span>
                    </td>
                    <td><span class="param-value">${hyperparams.n || 0}</span></td>
                    <td><span class="param-value">${hyperparams.k || 1}</span></td>
                    <td><span class="param-value">${hyperparams.sampling || 'T=1'}</span></td>
                    <td>
                        <span class="accuracy-badge">${formatPercent(accuracy)}</span>
                    </td>
                    <td><span class="stat-small">${eval.summary.total_rollouts || 0}</span></td>
                    <td><span class="stat-small">${formatPercent(eval.summary.eos_ratio || 0)}</span></td>
                </tr>
            `;
        }
        
        function renderTable() {
            const tbody = document.getElementById('evaluationsBody');
            tbody.innerHTML = '';
            
            sortedEvaluations.forEach((eval, index) => {
                // Find the original index for the detail view
                const originalIndex = evaluations.indexOf(eval);
                tbody.innerHTML += createEvaluationRow(eval, originalIndex);
            });
        }
        
        function getCellClass(originalColumn) {
            if (originalColumn === 'samples/chain_of_thought') {
                return 'cot-column';
            }
            if (originalColumn === 'samples/response') {
                return 'response-column';
            }
            return '';
        }
        
        function renderDetailTable() {
            const table = document.getElementById('detailTable');
            const columns = currentEvaluation.columns;
            const displayColumns = currentEvaluation.display_columns;
            
            // Create header
            let tableHtml = '<thead><tr>';
            columns.forEach((col, index) => {
                const displayCol = displayColumns[index];
                tableHtml += `<th onclick="sortDetailTable('${col}')" data-detail-column="${col}">${escapeHtml(displayCol)}<span class="sort-indicator"></span></th>`;
            });
            tableHtml += '</tr></thead><tbody>';
            
            // Create rows from sorted data
            sortedDetailData.forEach((row, rowIndex) => {
                tableHtml += '<tr>';
                columns.forEach(col => {
                    const value = row[col] !== null && row[col] !== undefined ? String(row[col]) : '';
                    const cellId = `cell-${rowIndex}-${col}`;
                    const cellClass = getCellClass(col);
                    tableHtml += `<td id="${cellId}" class="${cellClass}" onclick="showCellModal('${cellId}')" data-content="${escapeHtml(value).replace(/"/g, '&quot;')}">${escapeHtml(value)}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody>';
            
            table.innerHTML = tableHtml;
            
            // Update sort indicators
            if (currentDetailSort.column) {
                const currentHeader = document.querySelector(`[data-detail-column="${currentDetailSort.column}"]`);
                if (currentHeader) {
                    currentHeader.classList.add(currentDetailSort.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
                }
            }
        }
        
        function createDetailSummary(eval, index) {
            const accuracy = eval.summary.overall_accuracy || 0;
            const hyperparams = eval.hyperparams;
            const modelType = getModelType(hyperparams.model_path, hyperparams.mode);
            
            let difficultyHtml = '';
            if (eval.summary.difficulty_stats) {
                difficultyHtml = '<div class="difficulty-breakdown"><div class="difficulty-title">Accuracy by Difficulty:</div>';
                for (const [diff, stats] of Object.entries(eval.summary.difficulty_stats)) {
                    const accuracyPercent = stats.accuracy * 100;
                    difficultyHtml += `
                        <div class="difficulty-item">
                            <span>${diff}: ${accuracyPercent.toFixed(2)}% (n=${stats.count})</span>
                            <span class="accuracy-bar">
                                <span class="accuracy-fill" style="width: ${accuracyPercent}%"></span>
                            </span>
                        </div>
                    `;
                }
                difficultyHtml += '</div>';
            }
            
            return `
                <div class="detail-header">
                    <div class="model-name">${hyperparams.model_path || 'Unknown Model'} <span class="mode-badge ${getModeClass(hyperparams.model_path, hyperparams.mode)}">${modelType}</span></div>
                    <div class="accuracy-badge">${formatPercent(accuracy)}</div>
                </div>
                
                <div class="hyperparams-grid">
                    ${Object.entries(hyperparams).map(([key, value]) => `
                        <div class="param-item">
                            <div class="param-label">${key}</div>
                            <div class="param-value">${value}</div>
                        </div>
                    `).join('')}
                </div>
                
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-label">Total Rollouts</div>
                        <div class="stat-value">${eval.summary.total_rollouts || 0}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Unique Examples</div>
                        <div class="stat-value">${eval.summary.unique_examples || 0}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">EOS Token Rate</div>
                        <div class="stat-value">${formatPercent(eval.summary.eos_ratio || 0)}</div>
                    </div>
                </div>
                
                ${difficultyHtml}
            `;
        }
        
        function showMainPage() {
            document.getElementById('mainPage').style.display = 'block';
            document.getElementById('detailPage').style.display = 'none';
        }
        
        function showDetail(index) {
            currentEvaluation = evaluations[index];
            document.getElementById('mainPage').style.display = 'none';
            document.getElementById('detailPage').style.display = 'block';
            
            // Reset detail sort
            currentDetailSort = { column: null, direction: 'asc' };
            
            // Initialize sorted detail data
            sortedDetailData = [...currentEvaluation.data];
            
            // Update detail summary
            const summaryHtml = createDetailSummary(currentEvaluation, index);
            document.getElementById('detailSummary').innerHTML = summaryHtml;
            
            // Render detail table
            renderDetailTable();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function showCellModal(cellId) {
            const cell = document.getElementById(cellId);
            const content = cell.getAttribute('data-content');
            document.getElementById('modalContent').textContent = content.replace(/&quot;/g, '"');
            document.getElementById('cellModal').style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('cellModal').style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('cellModal');
            if (event.target == modal) {
                closeModal();
            }
        }
        
        // Initialize dashboard
        function init() {
            renderTable();
        }
        
        init();
    </script>
</body>
</html>'''
        
        return html_template
    
    def save_dashboard(self, output_path: str = "evaluation_dashboard.html"):
        """Process CSV files and save the dashboard HTML"""
        print(f"Processing CSV files from {self.folder_path}...")
        self.process_csv_files()
        
        if not self.evaluations:
            print("No evaluation CSV files found with n > 10!")
            return
        
        print(f"Found {len(self.evaluations)} evaluation files (n > 10)")
        
        html_content = self.generate_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation dashboard from CSV files')
    parser.add_argument('folder', type=str, help='Path to folder containing evaluation CSV files')
    parser.add_argument('--output', type=str, default='evaluation_dashboard.html', 
                       help='Output HTML file path (default: evaluation_dashboard.html)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist!")
        return
    
    dashboard = EvaluationDashboard(args.folder)
    dashboard.save_dashboard(args.output)
    
    print(f"\nDashboard created successfully!")
    print(f"Open {args.output} in your browser to view the dashboard.")


if __name__ == "__main__":
    main()