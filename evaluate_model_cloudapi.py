"""
EVALUATION SCRIPT - Cloud API Version
Uses OpenAI client (LLM Proxy) like chart_generator.py
Evaluates 100 queries and calculates runnable rate + chart-type accuracy
"""

import re
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from openai import OpenAI
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import time
from datetime import datetime
import os
import ast

# ==================== CONFIG ====================
# Khởi tạo OpenAI client với LLM Proxy
client = OpenAI(
    api_key="freellmapi-fe1244540184bfbbf1c2865f09560591d9f660a0315e499a",
    base_url="http://localhost:3001/v1"
)

# Paths
PROJECT_DIR = r"C:\Users\DELL\Desktop\python_viz"
CSV_QUERIES_PATH = os.path.join(PROJECT_DIR, "100_queries_from_automobile.csv")
CSV_AUTOMOBILE_PATH = r"C:\Users\DELL\Downloads\archive\Automobile_data.csv"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "test_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================
print("📁 Loading data...")
queries_df = pd.read_csv(CSV_QUERIES_PATH)
df = pd.read_csv(CSV_AUTOMOBILE_PATH, na_values=['?'], encoding='ISO-8859-1')

print(f"✅ Loaded {len(queries_df)} queries")
print(f"✅ Loaded automobile dataset with shape {df.shape}")

# ==================== MINI-RAG SYSTEM (từ chart_generator.py) ====================
class MiniRAGSystem:
    """Mini-RAG System để retrieval thông tin liên quan từ data."""
    def __init__(self, df):
        self.df = df
        self.columns = df.columns.tolist()
        self.column_descriptions = self._build_column_descriptions()
    
    def _build_column_descriptions(self):
        """Tạo mô tả chi tiết cho từng column."""
        descriptions = {}
        for col in self.columns:
            col_data = self.df[col]
            desc = {
                'name': col,
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                desc['stats'] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                }
            else:
                desc['sample_values'] = col_data.unique()[:5].tolist()
            
            descriptions[col] = desc
        return descriptions
    
    def _similarity_score(self, str1, str2):
        """Tính similarity giữa 2 chuỗi."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def retrieve_relevant_columns(self, query, top_k=5):
        """Tìm columns liên quan đến query."""
        query_lower = query.lower()
        scores = []
        
        for col in self.columns:
            name_score = self._similarity_score(col, query_lower)
            keyword_score = 0
            keywords = query_lower.split()
            for keyword in keywords:
                if len(keyword) > 2 and keyword in col.lower():
                    keyword_score += 0.3
            
            total_score = name_score + keyword_score
            scores.append((col, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_cols = [col for col, _ in scores[:top_k]]
        
        return relevant_cols
    
    def build_context(self, query):
        """Xây dựng RAG context dựa trên retrieved columns."""
        relevant_cols = self.retrieve_relevant_columns(query, top_k=5)
        
        context = f"""[MINI-RAG CONTEXT]
Query: {query}
Retrieved Columns: {', '.join(relevant_cols)}

Data Structure:
- Tổng rows: {len(self.df)}
- Tổng columns: {len(self.columns)}

Thông tin chi tiết các columns liên quan:
"""
        
        for col in relevant_cols:
            if col in self.column_descriptions:
                desc = self.column_descriptions[col]
                context += f"\n[{col}]"
                context += f"\n  - Type: {desc['dtype']}"
                context += f"\n  - Unique values: {desc['unique_count']}"
                context += f"\n  - Nulls: {desc['null_count']}"
                
                if 'stats' in desc:
                    stats = desc['stats']
                    context += f"\n  - Range: {stats['min']} to {stats['max']}"
                    context += f"\n  - Mean: {stats['mean']:.2f}"
                elif 'sample_values' in desc:
                    context += f"\n  - Sample values: {desc['sample_values']}"
        
        context += f"\n\nSample data (top 5 rows) của columns liên quan:"
        sample = self.df[relevant_cols].head(5)
        context += f"\n{sample.to_string()}"
        
        context += f"\n\nToàn bộ columns trong dataset:\n{', '.join(self.columns)}"
        
        return context

rag_system = MiniRAGSystem(df)

# ==================== CODE EXTRACTION & EXECUTION ====================
def extract_code_from_response(response_text):
    """Extract code from API response (chart_generator.py style)"""
    # Try to extract from ```python...```
    pattern = r"```python\s*\n(.*?)\n```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        # Try generic markdown
        pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        # Last resort: remove all markdown
        if "```" in response_text:
            code = response_text.replace("```", "").strip()
        else:
            code = response_text.strip()
    else:
        code = matches[0].strip()
    
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
        
        # Transform Ellipsis to df (handle ... in code)
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

# ==================== MAIN EVALUATION ====================
def evaluate_all_queries():
    """Evaluate all 100 queries using Cloud API"""
    results = []
    
    print("\n" + "="*80)
    print("🚀 STARTING CLOUD API EVALUATION")
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
            # Build RAG context
            data_context = rag_system.build_context(query)
            
            system_prompt = """Bạn là một chuyên gia Data Science. 
Khi được yêu cầu, hãy trả về CHỈ mã code Python thuần túy để vẽ biểu đồ.
Sử dụng Pandas, Matplotlib, hoặc Seaborn.
TUYỆT ĐỐI KHÔNG giải thích dài dòng, KHÔNG viết text ngoài code.
Bọc code trong cặp thẻ ```python và ```
Dữ liệu đã được load vào biến 'df', bạn có thể sử dụng trực tiếp."""
            
            full_prompt = f"""{data_context}

Yêu cầu: {query}"""
            
            # Call Cloud API
            print("    ⏳ Calling Cloud API...")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="auto",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            api_time = time.time() - start_time
            print(f"    ✅ Response ({api_time:.2f}s)")
            
            result_text = response.choices[0].message.content
            
            # Extract code
            code = extract_code_from_response(result_text)
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
    print("📊 EVALUATION RESULTS - CLOUD API")
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
    print("🎯 Starting Cloud API model evaluation...")
    print(f"📊 Test data: {CSV_QUERIES_PATH}")
    print(f"🌐 API: http://localhost:3001/v1")
    
    # Run evaluation
    results_df = evaluate_all_queries()
    
    # Calculate statistics
    calculate_statistics(results_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"evaluation_results_cloudapi_{timestamp}.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Results saved: {output_path}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"evaluation_summary_cloudapi_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLOUD API MODEL EVALUATION SUMMARY\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"API: http://localhost:3001/v1\n")
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
