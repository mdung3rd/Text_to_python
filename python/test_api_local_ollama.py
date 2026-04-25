
import json
import requests
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import ast
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

plt.rcParams['figure.max_open_warning'] = 50

OLLAMA_ENDPOINT_CANDIDATES = [
    "http://localhost:11434/api/generate",
    "http://127.0.0.1:11434/api/generate",
    "http://localhost:11434/v1/completions",
    "http://127.0.0.1:11434/v1/completions",
]
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120
OLLAMA_MAX_TOKENS = 384  
OLLAMA_TEMPERATURE = 0.2  
OLLAMA_TOP_P = 0.8

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def parse_ollama_response(response):
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


def ollama_local_complete(prompt, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT, max_tokens=OLLAMA_MAX_TOKENS, temperature=OLLAMA_TEMPERATURE, top_p=OLLAMA_TOP_P):
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
            last_error = RuntimeError(f"Empty response from Ollama endpoint: {endpoint}")
        except requests.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 404:
                continue
            last_error = http_err
        except requests.RequestException as req_err:
            last_error = req_err
            continue

    raise last_error or RuntimeError("Không thể kết nối đến Ollama local API. Kiểm tra lại server và endpoint.")


df = pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\Automobile_data.csv", na_values=['?'], encoding="ISO-8859-1")

sample_rows = df.head(3).to_string()
DATA_INFO = f"""
DataFrame information:
Rows: {len(df)}
Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
Sample data (first 3 rows):
{sample_rows}
"""

# ==================== MINI-RAG SYSTEM ====================
class SimpleRAG:
    """Mini RAG system để lưu trữ và truy xuất context liên quan"""
    def __init__(self):
        self.documents = []  # Danh sách tài liệu/context
        self.successful_codes = []  # Lưu trữ các đoạn code thành công
        self._build_initial_docs()
    
    def _build_initial_docs(self):
        """Xây dựng tài liệu ban đầu từ dataframe"""
        # 1. Thêm thông tin cơ bản về mỗi cột
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # Thêm thống kê cho cột numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = f"Column '{col}': {dtype}, {unique_count} unique values, {null_count} nulls. Min={df[col].min()}, Max={df[col].max()}, Mean={df[col].mean():.2f}"
            else:
                stats = f"Column '{col}': {dtype}, {unique_count} unique values, {null_count} nulls"
            
            self.documents.append({
                'type': 'column_info',
                'content': stats,
                'keywords': [col.lower(), dtype.lower()]
            })
        
        # 2. Thêm các ví dụ visualization phổ biến
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
        """Lưu trữ code thành công kèm prompt"""
        self.successful_codes.append({
            'prompt': prompt.lower(),
            'code': code,
            'keywords': self._extract_keywords(prompt)
        })
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa từ text"""
        # Loại bỏ các từ phổ biến
        stopwords = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 
                    'cái', 'các', 'của', 'để', 'với', 'có', 'là', 'từ', 'và'}
        words = re.findall(r'\b[a-z_-]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def retrieve(self, prompt: str, top_k: int = 3) -> str:
        """Tìm kiếm context liên quan đến prompt"""
        prompt_keywords = set(self._extract_keywords(prompt))
        
        # Tính điểm liên quan cho mỗi document
        scores = []
        for doc in self.documents:
            doc_keywords = set(doc.get('keywords', []))
            # Độ tương đồng: số từ khóa chung / tổng từ khóa
            overlap = len(prompt_keywords & doc_keywords)
            score = overlap + (0.5 if any(kw in prompt.lower() for kw in doc_keywords) else 0)
            scores.append((score, doc))
        
        # Sắp xếp theo điểm và lấy top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        relevant_docs = [doc for score, doc in scores[:top_k] if score > 0]
        
        # Thêm các successful codes liên quan
        code_scores = []
        for code_entry in self.successful_codes:
            overlap = len(set(code_entry['keywords']) & prompt_keywords)
            if overlap > 0:
                code_scores.append((overlap, code_entry))
        
        code_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Format output
        result = "\n--- RETRIEVED CONTEXT ---\n"
        
        if relevant_docs:
            result += "Relevant information:\n"
            for doc in relevant_docs:
                result += f"- {doc['content']}\n"
        
        if code_scores:
            result += "\nPrevious successful code examples:\n"
            for score, code_entry in code_scores[:2]:  # Top 2 code examples
                result += f"Example: {code_entry['code'][:200]}...\n"
        
        result += "--- END CONTEXT ---\n\n"
        return result if (relevant_docs or code_scores) else ""

# Khởi tạo RAG
rag = SimpleRAG()

# Tạo alias an toàn cho các tên cột không phải identifier Python (vd. có dấu '-')
def make_safe_alias(col):
    alias = re.sub(r'\W+', '_', col)
    if alias and alias[0].isdigit():
        alias = '_' + alias
    return alias

# Hàm kiểm tra xem prompt có liên quan đến visualization/data analysis hay không
def is_valid_prompt(prompt):
    """Kiểm tra xem prompt có chứa từ khóa liên quan đến visualization/data analysis"""
    visualization_keywords = [
        'plot', 'chart', 'histogram', 'graph', 'scatter', 'bar', 'line',
        'visualization', 'visualize', 'draw', 'show', 'display',
        'đồ thị', 'biểu đồ', 'vẽ', 'hiển thị'
    ]
    data_keywords = [
        'column', 'row', 'data', 'value', 'distribution', 'count', 'analysis',
        'analyze', 'relationship', 'correlation', 'group',
        'cột', 'dữ liệu', 'giá trị', 'phân phối', 'thống kê', 'phân tích'
    ]
    
    prompt_lower = prompt.lower().strip()
    
    # Kiểm tra độ dài - nếu quá ngắn (≤2 ký tự) thường không phải yêu cầu hợp lệ
    if len(prompt_lower) <= 2:
        return False
    
    # Kiểm tra xem có chứa từ khóa nào không
    all_keywords = visualization_keywords + data_keywords
    for keyword in all_keywords:
        if keyword in prompt_lower:
            return True
    
    # Kiểm tra xem prompt có chứa tên cột nào từ dataframe
    for col in df.columns:
        if col.lower() in prompt_lower:
            return True
    
    return False

for col in df.columns:
    safe = make_safe_alias(col)
    if safe != col and safe not in df.columns:
        df[safe] = df[col]


def run_prompt_loop():
    while True:
        prompt = input("Nhập yêu cầu (hoặc 'q' để thoát): ")
        if prompt.lower() == 'q':
            break
        
        # Xác thực prompt
        if not is_valid_prompt(prompt):
            print("❌ Yêu cầu không hợp lệ! Vui lòng nhập yêu cầu liên quan đến visualization hoặc phân tích dữ liệu.")
            continue

        # 🎯 Truy xuất context liên quan từ RAG
        rag_context = rag.retrieve(prompt, top_k=3)
        
        full_prompt = DATA_INFO + rag_context + f"""Write ONLY valid Python code (no surrounding markdown fences).
Use seaborn or matplotlib to create visualizations.
Use dataframe df (a pandas DataFrame).
When referring to columns with hyphens or spaces, use bracket notation like df['engine-size'] or the underscore alias like df['engine_size'].
IMPORTANT: End your code with plt.show() to display the plot.
Do not explain anything — only output runnable Python code.
Task: {prompt}
"""

        try:
            raw_output = ollama_local_complete(full_prompt)
            stderr_output = ""
        except Exception as e:
            print("Lỗi khi gọi Ollama local API:", e)
            continue
    
        # Cố gắng trích xuất khối code giữa ```...``` (bỏ qua ngôn ngữ nếu có)
        m = re.search(r"```(?:\w+)?\n(.*?)```", raw_output, re.DOTALL)
        if m:
            code = m.group(1).strip()
        else:
            # Nếu có ``` nhưng không khớp pattern, loại bỏ các ``` rồi fallback
            if "```" in raw_output:
                code = raw_output.replace("```", "").strip()
            else:
                code = raw_output.strip()
    
        # Xóa ANSI escape codes (ký tự không in được từ terminal colors)
        code = re.sub(r'\x1b\[[0-9;]*m', '', code)  # \x1b = ESC (U+001B)
        code = re.sub(r'\033\[[0-9;]*m', '', code)  # \033 = octal ESC
        # Fallback: xóa các ký tự kiểm soát ASCII khác (ngoại trừ newline, tab)
        code = ''.join(c for c in code if ord(c) >= 32 or c in '\n\r\t')
    
        print("\n--- Raw model output ---\n", raw_output)
        if stderr_output:
            print("\n--- Model stderr ---\n", stderr_output)
        print("\n--- Extracted code (after ANSI cleanup) ---\n", code)
    
        # Thay Ellipsis (``...``) thành biến `df` trong AST để tránh lỗi data=Ellipsis
        try:
            tree = ast.parse(code)
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
            print("SyntaxError khi compile code sinh ra:", se)
            print("Code (truncated):\n", code[:1000])
            continue
        except Exception as e:
            # Nếu AST xử lý thất bại, thử compile thẳng như fallback
            try:
                compiled = compile(code, "<string>", "exec")
            except SyntaxError as se2:
                print("SyntaxError khi compile code (fallback):", se2)
                print("Code (truncated):\n", code[:1000])
                continue
    
        try:
            exec(compiled, {"df": df, "plt": plt, "sns": sns, "pd": pd, "np": np})
            # Nếu code không gọi plt.show(), tự động gọi nếu có figure
            if plt.get_fignums():
                plt.show()
            # Xóa figure sau khi hiển thị để tránh tích lũy
            plt.close('all')
            
            # ✅ Lưu code thành công vào RAG để sử dụng lại
            rag.add_successful_code(prompt, code)
            print("✅ Code thực thi thành công! Đã lưu vào knowledge base.")
            
        except Exception as e:
            print("Lỗi khi chạy code:", e)
            plt.close('all')


if __name__ == "__main__":
    run_prompt_loop()
