
import subprocess
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import ast
import numpy as np

plt.rcParams['figure.max_open_warning'] = 50

df = pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\Automobile_data.csv", na_values=['?'], encoding="ISO-8859-1")
 
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

while True:
    prompt = input("Nhập yêu cầu (hoặc 'q' để thoát): ")
    if prompt.lower() == 'q':
        break
    
    # Xác thực prompt
    if not is_valid_prompt(prompt):
        print("❌ Yêu cầu không hợp lệ! Vui lòng nhập yêu cầu liên quan đến visualization hoặc phân tích dữ liệu.")
        continue

    # Mini RAG: Retrieve and include dataframe information to augment the prompt
    data_info = f"""
DataFrame Information:
Columns: {list(df.columns)}
Data types: {df.dtypes.to_dict()}
Sample data (first 5 rows):
{df.head().to_string()}
"""

    full_prompt = data_info + f"""
Write ONLY valid Python code (no surrounding markdown fences).
Use seaborn or matplotlib to create visualizations.
Use dataframe df (a pandas DataFrame).
When referring to columns with hyphens or spaces, use bracket notation like df['engine-size'] or the underscore alias like df['engine_size'].
IMPORTANT: End your code with plt.show() to display the plot.
Do not explain anything — only output runnable Python code.
Task: {prompt}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3", full_prompt],
        text=True,
        capture_output=True,
        encoding="utf-8"
    )
    raw_output = result.stdout
    stderr_output = result.stderr

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