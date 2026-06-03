"""
ROBUST EVALUATION SCRIPT - Based on test_api_local_ollama.py
Evaluates 100 queries and calculates runnable rate + chart-type accuracy
"""

import json
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import ast
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import time
from datetime import datetime
import os

# ==================== CONFIG ====================
OLLAMA_ENDPOINT_CANDIDATES = [
    "http://127.0.0.1:11434/v1/completions",
]
OLLAMA_MODEL = "llama3.2-lite"
OLLAMA_TIMEOUT = 120
OLLAMA_MAX_TOKENS = 384
OLLAMA_TEMPERATURE = 0.2
OLLAMA_TOP_P = 0.8

# Paths
PROJECT_DIR = r"C:\Users\DELL\Desktop\python_viz"
CSV_QUERIES_PATH = os.path.join(PROJECT_DIR, "100_queries_from_automobile.csv")
CSV_AUTOMOBILE_PATH = r"C:\Users\DELL\Downloads\archive\Automobile_data.csv"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "test_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================
print("📁 Loading data...")
queries_df = pd.read_csv(CSV_QUERIES_PATH)
df = pd.read_csv(CSV_AUTOMOBILE_PATH, na_values=['?'], encoding="ISO-8859-1")

print(f"✅ Loaded {len(queries_df)} queries")
print(f"✅ Loaded automobile dataset with shape {df.shape}")

# ==================== SESSION ====================
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# ==================== DATA INFO ====================
sample_rows = df.head(3).to_string()
DATA_INFO = f"""
DataFrame information:
Rows: {len(df)}
Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
Sample data (first 3 rows):
{sample_rows}
"""

# ==================== MINI-RAG SYSTEM (từ test_api_local_ollama.py) ====================
class SimpleRAG:
    """Mini RAG system để lưu trữ và truy xuất context liên quan"""
    def __init__(self):
        self.documents = []
        self.successful_codes = []
        self._build_initial_docs()
    
    def _build_initial_docs(self):
        """Xây dựng tài liệu ban đầu từ dataframe"""
        # 1. Thêm thông tin cơ bản về mỗi cột
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = f"Column '{col}': {dtype}, {unique_count} unique values, {null_count} nulls. Min={df[col].min()}, Max={df[col].max()}, Mean={df[col].mean():.2f}"
            else:
                stats = f"Column '{col}': {dtype}, {unique_count} unique values, {null_count} nulls"
            
            self.documents.append({
                'type': 'column_info',
                'content': stats,
                'keywords': [col.lower(), dtype.lower()]
            })
        
        # 2. Visualization examples
        examples = [
            {
                'type': 'example',
                'content': "For numeric distribution: plt.hist(df['column'], bins=30); plt.xlabel('column'); plt.ylabel('frequency'); plt.title('Distribution')",
                'keywords': ['histogram', 'distribution', 'numeric', 'hist']
            },
            {
                'type': 'example',
                'content': "For scatter plot: plt.scatter(df['x_col'], df['y_col']); plt.xlabel('x_col'); plt.ylabel('y_col'); plt.title('Relationship')",
                'keywords': ['scatter', 'relationship', 'correlation', 'plot']
            },
            {
                'type': 'example',
                'content': "For bar plot: df['category'].value_counts().plot(kind='bar'); plt.title('Counts'); plt.xlabel('category'); plt.ylabel('count')",
                'keywords': ['bar', 'count', 'categorical', 'value_counts']
            },
            {
                'type': 'example',
                'content': "For pie chart: counts = df['category'].value_counts(); plt.figure(figsize=(8, 6)); plt.pie(counts, labels=counts.index, autopct='%1.1f%%'); plt.title('Distribution')",
                'keywords': ['pie', 'pie chart', 'percentage', 'distribution', 'autopct']
            },
            {
                'type': 'example',
                'content': "For correlation heatmap: sns.heatmap(df.corr(), annot=True, cmap='coolwarm'); plt.title('Correlation Matrix')",
                'keywords': ['correlation', 'heatmap', 'corr', 'matrix']
            },
            {
                'type': 'example',
                'content': "For box plot: sns.boxplot(data=df, y='numeric_col'); plt.title('Box Plot')",
                'keywords': ['box', 'boxplot', 'outlier', 'distribution', 'quartile']
            }
        ]
        self.documents.extend(examples)
    
    def add_successful_code(self, prompt: str, code: str):
        """Lưu trữ code thành công"""
        self.successful_codes.append({
            'prompt': prompt.lower(),
            'code': code,
            'keywords': self._extract_keywords(prompt)
        })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa từ text"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 
                    'cái', 'các', 'của', 'để', 'với', 'có', 'là', 'từ', 'và'}
        words = re.findall(r'\b[a-z_-]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def retrieve(self, prompt: str, top_k: int = 3) -> str:
        """Tìm kiếm context liên quan"""
        prompt_keywords = set(self._extract_keywords(prompt))
        
        scores = []
        for doc in self.documents:
            doc_keywords = set(doc.get('keywords', []))
            overlap = len(prompt_keywords & doc_keywords)
            score = overlap + (0.5 if any(kw in prompt.lower() for kw in doc_keywords) else 0)
            scores.append((score, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        relevant_docs = [doc for score, doc in scores[:top_k] if score > 0]
        
        code_scores = []
        for code_entry in self.successful_codes:
            overlap = len(set(code_entry['keywords']) & prompt_keywords)
            if overlap > 0:
                code_scores.append((overlap, code_entry))
        
        code_scores.sort(key=lambda x: x[0], reverse=True)
        
        result = "\n--- RETRIEVED CONTEXT ---\n"
        if relevant_docs:
            result += "Relevant information:\n"
            for doc in relevant_docs:
                result += f"- {doc['content']}\n"
        
        if code_scores:
            result += "\nPrevious successful code examples:\n"
            for score, code_entry in code_scores[:2]:
                result += f"Example: {code_entry['code'][:200]}...\n"
        
        result += "--- END CONTEXT ---\n\n"
        return result if (relevant_docs or code_scores) else ""

rag = SimpleRAG()

# ==================== API FUNCTIONS (từ test_api_local_ollama.py) ====================
def parse_ollama_response(response):
    """Parse Ollama response - robust parsing"""
    text = response.text.strip()
    if not text:
        return ""

    try:
        return response.json()
    except ValueError:
        lines = [line for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
        return text

def extract_model_text(data):
    """Extract text from model response"""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "".join(extract_model_text(item) for item in data)
    if isinstance(data, dict):
        if "response" in data:
            return str(data["response"])
        if "choices" in data:
            choices = data["choices"]
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    return first_choice.get("text", first_choice.get("content", ""))
                return str(first_choice)
        if "output" in data:
            output = data["output"]
            if isinstance(output, list):
                return "".join(
                    item.get("content", "") if isinstance(item, dict) else str(item)
                    for item in output
                )
            return str(output)
        if "text" in data:
            return str(data["text"])
    return str(data)

def ollama_local_complete(prompt, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT, 
                          max_tokens=OLLAMA_MAX_TOKENS, temperature=OLLAMA_TEMPERATURE, 
                          top_p=OLLAMA_TOP_P):
    """Call Ollama local API"""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_error = None
    for endpoint in OLLAMA_ENDPOINT_CANDIDATES:
        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = parse_ollama_response(response)
            text = extract_model_text(data).strip()
            if text:
                return text
            last_error = RuntimeError(f"Empty response from endpoint: {endpoint}")
        except requests.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 404:
                continue
            last_error = http_err
        except requests.RequestException as req_err:
            last_error = req_err
            continue

    raise last_error or RuntimeError("Cannot connect to Ollama local API")

# ==================== CODE EXTRACTION & EXECUTION ====================
def extract_code_from_response(raw_output):
    """Extract code from model response"""
    # Try to extract from markdown fences
    m = re.search(r"```(?:\w+)?\n(.*?)```", raw_output, re.DOTALL)
    if m:
        code = m.group(1).strip()
    else:
        if "```" in raw_output:
            code = raw_output.replace("```", "").strip()
        else:
            code = raw_output.strip()
    
    # Remove ANSI escape codes
    code = re.sub(r'\x1b\[[0-9;]*m', '', code)
    code = re.sub(r'\033\[[0-9;]*m', '', code)
    code = ''.join(c for c in code if ord(c) >= 32 or c in '\n\r\t')
    
    return code

def execute_code_safely(code):
    """Execute code safely with AST transformation (robust approach)"""
    try:
        # Parse and transform AST
        tree = ast.parse(code)
        
        # Transform Ellipsis to df
        class EllipsisToDf(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value is Ellipsis:
                    return ast.copy_location(ast.Name(id='df', ctx=ast.Load()), node)
                return node
            def visit_Ellipsis(self, node):
                return ast.copy_location(ast.Name(id='df', ctx=ast.Load()), node)
        
        transformer = EllipsisToDf()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        compiled = compile(tree, "<string>", "exec")
        
    except SyntaxError as se:
        return False, f"SyntaxError: {str(se)}", False
    except Exception as e:
        # Fallback: try compile directly
        try:
            compiled = compile(code, "<string>", "exec")
        except SyntaxError as se2:
            return False, f"SyntaxError: {str(se2)}", False
    
    try:
        plt.close('all')
        exec(compiled, {"df": df, "plt": plt, "sns": sns, "pd": pd, "np": np})
        has_figure = len(plt.get_fignums()) > 0
        plt.close('all')
        return True, None, has_figure
    except Exception as e:
        plt.close('all')
        return False, f"{type(e).__name__}: {str(e)}", False

# ==================== CHART TYPE DETECTION ====================
def detect_chart_type_from_code(code):
    """Detect chart type from code - improved chart type detection with Seaborn support"""
    code_lower = code.lower()
    
    # Check for specific chart types with more flexible patterns
    detected_types = []
    
    # Pie charts
    if re.search(r'plt\.pie\(|\.pie\(', code_lower):
        detected_types.append('Pie')
    
    # Bar charts - matplotlib and seaborn
    if (re.search(r'plt\.bar\(|sns\.barplot\(', code_lower) or 
        re.search(r'\.bar\(', code_lower) or
        re.search(r'\.plot\([^)]*kind\s*=\s*["\']?bar["\']?', code_lower)):
        detected_types.append('Bar')
    
    # Scatter plots - matplotlib and seaborn
    if re.search(r'plt\.scatter\(|sns\.scatterplot\(|\.scatter\(', code_lower):
        detected_types.append('Scatter')
    
    # Histograms - matplotlib and seaborn
    if re.search(r'plt\.hist\(|sns\.histplot\(|\.hist\(|\.hist\s*\(', code_lower):
        detected_types.append('Histogram')
    
    # Boxplots - matplotlib and seaborn
    if re.search(r'plt\.boxplot\(|sns\.boxplot\(|\.boxplot\(', code_lower):
        detected_types.append('Boxplot')
    
    # Line plots - only if NOT a bar chart (to avoid false positives)
    # Check for plt.plot() but NOT kind='bar'
    if (re.search(r'plt\.plot\(|\.plot\(', code_lower) and 
        not re.search(r'\.plot\([^)]*kind\s*=\s*["\']?bar["\']?', code_lower)):
        detected_types.append('Line')
    
    # Heatmaps
    if re.search(r'sns\.heatmap\(|\.heatmap\(', code_lower):
        detected_types.append('Heatmap')
    
    # Return first detected type (remove duplicates)
    if detected_types:
        seen = set()
        unique_types = []
        for t in detected_types:
            if t not in seen:
                unique_types.append(t)
                seen.add(t)
        return unique_types[0] if unique_types else None
    
    return None

# ==================== MAKE SAFE ALIAS ====================
def make_safe_alias(col):
    """Create safe alias for columns with special chars"""
    alias = re.sub(r'\W+', '_', col)
    if alias and alias[0].isdigit():
        alias = '_' + alias
    return alias

# Create aliases
for col in df.columns:
    safe = make_safe_alias(col)
    if safe != col and safe not in df.columns:
        df[safe] = df[col]

# ==================== MAIN EVALUATION ====================
def evaluate_all_queries():
    """Evaluate all 100 queries"""
    results = []
    
    print("\n" + "="*80)
    print("🚀 STARTING ROBUST EVALUATION")
    print("="*80)
    
    total = len(queries_df)
    
    for idx, row in queries_df.iterrows():
        query_id = idx + 1
        query = row['Query']
        expected_chart = row['Expected_Chart_Type']
        required_cols = row.get('Required_Columns', '')
        difficulty = row.get('Difficulty', '')
        
        print(f"\n[{query_id}/{total}] 📝 Query: {query[:60]}...")
        print(f"    Expected: {expected_chart}")
        
        try:
            # Retrieve RAG context
            rag_context = rag.retrieve(query, top_k=3)
            
            full_prompt = DATA_INFO + rag_context + f"""Write ONLY valid Python code (no surrounding markdown fences).
Use seaborn or matplotlib to create visualizations.
Use dataframe df (a pandas DataFrame).
When referring to columns with hyphens or spaces, use bracket notation like df['engine-size'] or the underscore alias like df['engine_size'].
IMPORTANT: End your code with plt.show() to display the plot.
Rule 4: Always explicitly create a new figure at the beginning of your code using plt.figure(figsize=(8, 6)) to ensure a fresh, correctly sized canvas.
Rule 5: CRITICAL - If the task explicitly requests a pie chart, MUST use plt.pie() with autopct parameter for percentage labels. Do NOT use bar chart instead of pie chart.
Rule 6: CRITICAL - If the task explicitly requests a bar chart, use bar charts. If requesting pie chart, use ONLY pie charts.
Do not explain anything — only output runnable Python code.
Task: {query}
"""
            
            # Call API
            print("    ⏳ Calling API...")
            start_time = time.time()
            raw_output = ollama_local_complete(full_prompt)
            api_time = time.time() - start_time
            print(f"    ✅ Response ({api_time:.2f}s)")
            
            # Extract code
            code = extract_code_from_response(raw_output)
            print(f"    📄 Code: {len(code)} chars")
            
            # Detect chart type
            detected_chart = detect_chart_type_from_code(code)
            print(f"    🎯 Detected: {detected_chart}")
            
            # Execute
            success, error_msg, has_figure = execute_code_safely(code)
            if success:
                print(f"    ✅ Runnable")
            else:
                print(f"    ❌ Error: {error_msg[:60]}")
            
            # Check accuracy
            chart_match = detected_chart == expected_chart if detected_chart else False
            
            result = {
                'query_id': query_id,
                'query': query,
                'expected_chart': expected_chart,
                'detected_chart': detected_chart,
                'chart_match': 'Yes' if chart_match else 'No',
                'runnable': 'Yes' if success else 'No',
                'error': error_msg or '',
                'api_time': f"{api_time:.2f}",
                'code_length': len(code),
                'difficulty': difficulty
            }
            
            results.append(result)
            
            # Add to RAG if successful
            if success:
                rag.add_successful_code(query, code)
        
        except Exception as e:
            print(f"    💥 Error: {str(e)[:60]}")
            result = {
                'query_id': query_id,
                'query': query,
                'expected_chart': expected_chart,
                'detected_chart': None,
                'chart_match': 'No',
                'runnable': 'No',
                'error': str(e),
                'api_time': '',
                'code_length': 0,
                'difficulty': difficulty
            }
            results.append(result)
    
    return pd.DataFrame(results)

# ==================== STATISTICS ====================
def calculate_statistics(results_df):
    """Calculate and print statistics"""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    total = len(results_df)
    runnable_count = (results_df['runnable'] == 'Yes').sum()
    runnable_rate = (runnable_count / total) * 100 if total > 0 else 0
    
    chart_match_count = (results_df['chart_match'] == 'Yes').sum()
    chart_accuracy = (chart_match_count / total) * 100 if total > 0 else 0
    
    runnable_df = results_df[results_df['runnable'] == 'Yes']
    runnable_chart_match = (runnable_df['chart_match'] == 'Yes').sum()
    conditional_accuracy = (runnable_chart_match / len(runnable_df)) * 100 if len(runnable_df) > 0 else 0
    
    # By chart type
    chart_stats = {}
    for chart_type in results_df['expected_chart'].unique():
        subset = results_df[results_df['expected_chart'] == chart_type]
        chart_stats[chart_type] = {
            'total': len(subset),
            'runnable': (subset['runnable'] == 'Yes').sum(),
            'correct': (subset['chart_match'] == 'Yes').sum(),
        }
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total queries:          {total}")
    print(f"   Runnable:               {runnable_count}/{total} ({runnable_rate:.1f}%)")
    print(f"   Chart type correct:     {chart_match_count}/{total} ({chart_accuracy:.1f}%)")
    print(f"   Accuracy (runnable):    {runnable_chart_match}/{len(runnable_df)} ({conditional_accuracy:.1f}%)")
    
    print(f"\n📊 BY CHART TYPE:")
    for chart_type in sorted(chart_stats.keys()):
        stats = chart_stats[chart_type]
        runnable_pct = (stats['runnable'] / stats['total']) * 100
        correct_pct = (stats['correct'] / stats['total']) * 100
        print(f"   {chart_type:12} | Count: {stats['total']:2} | Runnable: {stats['runnable']:2} ({runnable_pct:5.1f}%) | Correct: {stats['correct']:2} ({correct_pct:5.1f}%)")
    
    print(f"\n⏱️  BY DIFFICULTY:")
    for difficulty in sorted(results_df['difficulty'].unique()):
        subset = results_df[results_df['difficulty'] == difficulty]
        runnable_count_diff = (subset['runnable'] == 'Yes').sum()
        correct_count_diff = (subset['chart_match'] == 'Yes').sum()
        runnable_pct = (runnable_count_diff / len(subset)) * 100
        correct_pct = (correct_count_diff / len(subset)) * 100
        print(f"   {difficulty:12} | Count: {len(subset):2} | Runnable: {runnable_count_diff:2} ({runnable_pct:5.1f}%) | Correct: {correct_count_diff:2} ({correct_pct:5.1f}%)")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("🎯 Starting robust model evaluation...")
    
    # Run evaluation
    results_df = evaluate_all_queries()
    
    # Calculate statistics
    calculate_statistics(results_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"evaluation_results_robust_{timestamp}.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Results saved: {output_path}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"evaluation_summary_robust_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ROBUST MODEL EVALUATION SUMMARY\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        total = len(results_df)
        runnable_count = (results_df['runnable'] == 'Yes').sum()
        runnable_rate = (runnable_count / total) * 100 if total > 0 else 0
        chart_match_count = (results_df['chart_match'] == 'Yes').sum()
        chart_accuracy = (chart_match_count / total) * 100 if total > 0 else 0
        
        f.write(f"Total Queries:          {total}\n")
        f.write(f"Runnable Count:         {runnable_count}\n")
        f.write(f"Runnable Rate:          {runnable_rate:.2f}%\n")
        f.write(f"Chart Type Correct:     {chart_match_count}\n")
        f.write(f"Chart Type Accuracy:    {chart_accuracy:.2f}%\n")
    
    print(f"✅ Summary saved: {summary_path}")
    print("\n🎉 Evaluation complete!")
