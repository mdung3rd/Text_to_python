
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

plt.rcParams['figure.max_open_warning'] = 50

OLLAMA_ENDPOINT_CANDIDATES = [
    "http://localhost:11434/api/generate",
    "http://127.0.0.1:11434/api/generate",
    "http://localhost:11434/v1/completions",
    "http://127.0.0.1:11434/v1/completions",
]
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120
OLLAMA_MAX_TOKENS = 256  # Giảm từ 384 để nhanh hơn với model nhỏ
OLLAMA_TEMPERATURE = 0.2  # Tăng từ 0.1 để sáng tạo hơn
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

        full_prompt = DATA_INFO + f"""
Write ONLY valid Python code (no surrounding markdown fences).
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
        except Exception as e:
            print("Lỗi khi chạy code:", e)
            plt.close('all')


if __name__ == "__main__":
    run_prompt_loop()
