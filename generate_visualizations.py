import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

# Create output directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/report', exist_ok=True)

# Load result files
benchmarks = ['mmlu_logic', 'hellaswag', 'arc', 'mmlu_math']
results = {}
for benchmark in benchmarks:
    result_file = f'results/gemini_{benchmark}.json'
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results[benchmark] = json.load(f)

# Create comparison dataframe
metrics_data = []
for benchmark, data in results.items():
    metrics_data.append({
        'Benchmark': benchmark,
        'Accuracy': data['accuracy'],
        'Logical Consistency': data['merit_metrics'].get('logical_consistency', 0),
        'Samples': data['samples']
    })
df = pd.DataFrame(metrics_data)

# Accuracy vs Logical Consistency Plot
plt.figure(figsize=(10, 6))
x = np.arange(len(df))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, df['Accuracy'], width, label='Accuracy')
rects2 = ax.bar(x + width/2, df['Logical Consistency'], width, label='Logical Consistency')
ax.set_ylabel('Score')
ax.set_title('MERIT: Accuracy vs Logical Consistency Across Benchmarks')
ax.set_xticks(x)
ax.set_xticklabels(df['Benchmark'])
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('results/plots/merit_comparison.png')
plt.close()

# Error Analysis Plot
error_data = []
for benchmark, data in results.items():
    correct = sum(1 for ex in data['results'] if ex['correct'])
    incorrect = len(data['results']) - correct
    error_data.append({
        'Benchmark': benchmark,
        'Correct': correct,
        'Incorrect': incorrect
    })
error_df = pd.DataFrame(error_data)
plt.figure(figsize=(10, 6))
x = np.arange(len(error_df))
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, error_df['Correct'], width, label='Correct')
ax.bar(x + width/2, error_df['Incorrect'], width, label='Incorrect')
ax.set_ylabel('Number of Examples')
ax.set_title('Correct vs Incorrect Predictions by Benchmark')
ax.set_xticks(x)
ax.set_xticklabels(error_df['Benchmark'])
ax.legend()
plt.tight_layout()
plt.savefig('results/plots/error_analysis.png')
plt.close()

# Generate HTML Report
html_content = '<html><head><title>MERIT Evaluation Report</title>'
html_content += '<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;width:100%;} th,td{padding:8px;text-align:left;border:1px solid #ddd;} th{background-color:#f2f2f2;} img{max-width:100%;}</style>'
html_content += '</head><body>'
html_content += '<h1>MERIT: Multi-dimensional Evaluation of Reasoning in Transformers</h1>'
html_content += '<h2>Evaluation Summary</h2>'

# Summary Table
html_content += '<table><tr><th>Benchmark</th><th>Samples</th><th>Accuracy</th><th>Logical Consistency</th><th>Execution Time (s)</th></tr>'
for benchmark, data in results.items():
    html_content += f'<tr><td>{benchmark}</td><td>{data["samples"]}</td><td>{data["accuracy"]:.4f}</td><td>{data["merit_metrics"].get("logical_consistency", 0):.4f}</td><td>{data["execution_time"]:.2f}</td></tr>'
html_content += '</table>'

# Visualizations
html_content += '<h2>Visualizations</h2>'
html_content += '<h3>Accuracy vs Logical Consistency</h3>'
html_content += '<img src="../plots/merit_comparison.png" alt="Accuracy vs Consistency">'
html_content += '<h3>Error Analysis</h3>'
html_content += '<img src="../plots/error_analysis.png" alt="Error Analysis">'

# Benchmark Details
for benchmark, data in results.items():
    html_content += f'<h2>{benchmark} Detailed Results</h2>'
    html_content += f'<p>Model: {data["model"]}, Adapter: {data["adapter"]}</p>'
    
    # Sample Results
    html_content += '<h3>Sample Evaluation Examples</h3>'
    html_content += '<table><tr><th>Example</th><th>Correct?</th><th>Prediction</th><th>Actual</th><th>Logical Consistency</th></tr>'
    for i, example in enumerate(data['results'][:5]):
        prediction = str(example['prediction']).replace('<', '&lt;').replace('>', '&gt;')
        actual = str(example['actual_label']).replace('<', '&lt;').replace('>', '&gt;')
        logic_score = 'N/A'
        for metric_name, metric_result in example['metrics'].items():
            if metric_name == 'logical_consistency' and isinstance(metric_result, dict) and 'score' in metric_result:
                logic_score = f'{metric_result["score"]:.4f}'
        html_content += f'<tr><td>{i+1}</td><td>{("✓" if example["correct"] else "✗")}</td><td>{prediction}</td><td>{actual}</td><td>{logic_score}</td></tr>'
    html_content += '</table>'
    
    # Case Studies
    html_content += '<h3>Case Studies</h3>'
    for i, example in enumerate([ex for ex in data['results'] if not ex['correct']][:3]):  # 3 incorrect examples
        logic_score = 'N/A'
        for metric_name, metric_result in example['metrics'].items():
            if metric_name == 'logical_consistency' and isinstance(metric_result, dict) and 'score' in metric_result:
                logic_score = f'{metric_result["score"]:.4f}'
        html_content += f'<h4>Example {i+1} (Incorrect, Logical Consistency: {logic_score})</h4>'
        html_content += f'<p><strong>Prompt:</strong> {example["prompt"][:200].replace("<", "&lt;").replace(">", "&gt;")}...</p>'
        html_content += f'<p><strong>Response:</strong> {example["response"][:200].replace("<", "&lt;").replace(">", "&gt;")}...</p>'
        html_content += f'<p><strong>Correct Answer:</strong> {example["reference"].replace("<", "&lt;").replace(">", "&gt;")}</p>'
        html_content += f'<p><strong>Prediction:</strong> {example["prediction"]}, Actual: {example["actual_label"]}, Correct: No</p>'
        html_content += f'<p><strong>Analysis:</strong> This example shows {"coherent but incorrect reasoning" if float(logic_score or 0) > 0.9 else "incorrect reasoning with potential inconsistencies"}.</p>'

html_content += '</body></html>'

# Write HTML Report
with open('results/report/merit_evaluation_report.html', 'w') as f:
    f.write(html_content)

# Print Summary
print('\nMERIT Evaluation Results Summary:')
print('-' * 80)
print(df.to_string(index=False))
print('-' * 80)
print('\nResults and visualizations saved in results/plots/')
print('\nHTML report generated at results/report/merit_evaluation_report.html')
