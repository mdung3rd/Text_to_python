import re
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from openai import OpenAI
from difflib import SequenceMatcher

# 9Router combo model đã được cấu hình trên local web.
# Model "Cline" tự fallback theo cấu hình của 9Router, không cần fallback thủ công trong code.
SELECTED_MODEL = "Cline"

# API key cho 9Router local
API_KEY = os.getenv("NINEROUTER_API_KEY", "sk-b3062a96fb131ad4-z8mnz7-f6b64e72")

# API configuration for 9router
API_CONFIG = {
    "base_url": "http://localhost:20128/v1",  # Endpoint của 9router
}

# Initialize client for 9router
client = OpenAI(
    api_key=API_KEY,
    base_url=API_CONFIG["base_url"]
)

# Đường dẫn file CSV
CSV_PATH = r"C:\Users\DELL\Downloads\archive\Automobile_data.csv"

# Compile regex patterns một lần để trích xuất code nhanh hơn ở mỗi request
CODE_BLOCK_PATTERNS = [
    re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```python(.*?)```", re.DOTALL),
    re.compile(r"```\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```(.*?)```", re.DOTALL),
]


def extract_python_code(result_text):
    """
    Trích xuất code Python từ response của LLM.
    Giữ nguyên thứ tự ưu tiên pattern như logic cũ nhưng tránh compile regex lặp lại.
    """
    for pattern in CODE_BLOCK_PATTERNS:
        matches = pattern.findall(result_text)
        if matches:
            return matches[0].strip()

    return None

class MiniRAGSystem:
    """
    Mini-RAG System để retrieval thông tin liên quan từ data.
    Sử dụng keyword matching để tìm columns và statistics liên quan đến query.
    """
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
            
            # Thêm statistics cho numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                desc['stats'] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                }
            else:
                # Lấy top values cho categorical columns
                desc['sample_values'] = col_data.unique()[:5].tolist()
            
            descriptions[col] = desc
        return descriptions
    
    def _similarity_score(self, str1, str2):
        """Tính similarity giữa 2 chuỗi."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def retrieve_relevant_columns(self, query, top_k=5):
        """
        Tìm columns liên quan đến query dựa trên keyword matching.
        
        Args:
            query (str): User query
            top_k (int): Số columns trả về
        
        Returns:
            list: Danh sách columns liên quan
        """
        query_lower = query.lower()
        scores = []
        
        for col in self.columns:
            # Tính similarity với column name
            name_score = self._similarity_score(col, query_lower)
            
            # Tìm keyword matches trong query
            keyword_score = 0
            keywords = query_lower.split()
            for keyword in keywords:
                if len(keyword) > 2 and keyword in col.lower():
                    keyword_score += 0.3
            
            total_score = name_score + keyword_score
            scores.append((col, total_score))
        
        # Sort và lấy top K
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_cols = [col for col, _ in scores[:top_k]]
        
        return relevant_cols
    
    def build_context(self, query):
        """
        Xây dựng RAG context dựa trên retrieved columns.
        
        Args:
            query (str): User query
        
        Returns:
            str: Context để gửi đến LLM
        """
        # Retrieve relevant columns
        relevant_cols = self.retrieve_relevant_columns(query, top_k=5)
        
        context = f"""[MINI-RAG CONTEXT]
Query: {query}
Retrieved Columns: {', '.join(relevant_cols)}

Data Structure:
- Tổng rows: {len(self.df)}
- Tổng columns: {len(self.columns)}

Thông tin chi tiết các columns liên quan:
"""
        
        # Lấy description cho các columns liên quan
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
        
        # Thêm sample data của relevant columns
        context += f"\n\nSample data (top 5 rows) của columns liên quan:"
        sample = self.df[relevant_cols].head(5)
        context += f"\n{sample.to_string()}"
        
        # Thêm toàn bộ danh sách columns để model biết
        context += f"\n\nToàn bộ columns trong dataset:\n{', '.join(self.columns)}"
        
        return context



def load_and_analyze_data(csv_path):
    """
    Đọc file CSV và khởi tạo Mini-RAG System.
    
    Args:
        csv_path (str): Đường dẫn tới file CSV
    
    Returns:
        tuple: (df, rag_system)
    """
    try:
        df = pd.read_csv(csv_path, na_values=['?'], encoding='ISO-8859-1')
        print(f"[✓] Đã load dữ liệu: {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        rag_system = MiniRAGSystem(df)
        return df, rag_system
    
    except FileNotFoundError:
        print(f"[✗] Không tìm thấy file: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[✗] Lỗi đọc file: {e}")
        sys.exit(1)

def is_valid_chart_request(prompt):
    """
    Kiểm tra xem user prompt có liên quan đến việc vẽ biểu đồ hay không.
    
    Args:
        prompt (str): User prompt
    
    Returns:
        bool: True nếu hợp lệ, False nếu không
    """
    chart_keywords = [
        'chart', 'plot', 'graph', 'biểu đồ', 'vẽ', 'visualization', 'visualize',
        'histogram', 'scatter', 'line', 'bar', 'pie', 'box', 'heatmap',
        'histogram', 'lines', 'bars', 'pies', 'boxes',
        'đồ thị', 'hình vẽ', 'draw', 'show', 'display',
        'relationship', 'compare', 'analysis', 'analyze', 'trend',
        'distribution', 'correlation', 'pattern', 'trend', 'visual'
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in chart_keywords)

def generate_and_run_chart(user_prompt, df, rag_system):
    """
    Hàm nhận user_prompt, dùng Mini-RAG để retrieve context liên quan,
    gọi API LLM để sinh code Python vẽ biểu đồ, trích xuất và thực thi.
    
    Args:
        user_prompt (str): Yêu cầu từ người dùng
        df (pd.DataFrame): DataFrame của dữ liệu
        rag_system (MiniRAGSystem): Mini-RAG system instance
    """
    
    # System Prompt bắt buộc AI đóng vai chuyên gia Data Science với các hạn chế bảo mật
    system_prompt = """Bạn là một chuyên gia Data Science. 

QUY TẮC NGHIÊM NGẶT:
1. CHỈ sinh code Python thuần túy để VẼ BIỂU ĐỒ, KHÔNG làm gì khác
2. Dữ liệu đã được load vào biến 'df' - CHỈ sử dụng biến này, không đọc file khác
3. CHỈ sử dụng các thư viện sau: pandas, matplotlib, seaborn, numpy
4. KHÔNG sử dụng: os, sys, subprocess, requests, urllib, open, file operations
5. KHÔNG giải thích, KHÔNG viết text ngoài code
6. KHÔNG cài đặt package, KHÔNG import thư viện ngoài danh sách trên
7. Code PHẢI được bọc trong ```python và ```
8. Nếu không thể vẽ biểu đồ, trả về thông báo rõ ràng trong code

BẮT ĐẦU CODE:"""
    
    # ==== MINI-RAG RETRIEVAL ====
    print("[*] Mini-RAG: Retrieving relevant context...")
    data_context = rag_system.build_context(user_prompt)
    print("[✓] Context retrieved từ RAG system\n")
    
    # Kết hợp context dữ liệu và user prompt
    full_prompt = f"""{data_context}

Yêu cầu: {user_prompt}

LƯU Ý: Chỉ trả về code Python thuần túy, không giải thích gì thêm."""
    
    try:
        # Gọi API sử dụng OpenAI SDK
        print("[*] Đang gọi API 9Router...")
        response = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        print("[✓] Nhận response từ API\n")
        
        # Trích xuất code từ response
        extracted_code = extract_python_code(result_text)
        if not extracted_code:
            print("[✗] Lỗi: Không tìm thấy code trong response")
            print(f"[DEBUG] Response nhận được:\n{result_text}")
            return
        
        # In đoạn code đã bóc tách ra console
        print("=" * 60)
        print("[CODE ĐƯỢC TRÍCH XUẤT]")
        print("=" * 60)
        print(extracted_code)
        print("=" * 60 + "\n")
        
        # CẢNH BÁO: Chạy code do AI sinh ra có thể không an toàn
        # Đây là project demo nên vẫn cho phép chạy, nhưng trong production cần validation
        print("[*] Đang thực thi code...\n")
        try:
            exec_globals = {
                'df': df,  # DataFrame chứa dữ liệu
                'pd': pd,  # Pandas library
                'plt': plt,  # Matplotlib pyplot đã import sẵn
                'sns': sns,  # Seaborn library đã import sẵn
                'np': np,  # Numpy library đã import sẵn
            }
            exec(extracted_code, exec_globals)
            print("\n[✓] Code thực thi thành công!")
        except SyntaxError as e:
            print(f"[✗] Lỗi cú pháp (SyntaxError): {e}")
            print(f"    Dòng {e.lineno}: {e.text}")
        except Exception as e:
            print(f"[✗] Lỗi khi thực thi: {type(e).__name__}: {e}")
    
    except Exception as e:
        print(f"[✗] Lỗi kết nối API: {e}")


if __name__ == "__main__":
    # Xử lý argument cho file CSV
    csv_path = CSV_PATH  # Mặc định
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"[*] Sử dụng file CSV từ argument: {csv_path}")
    else:
        print(f"[*] Sử dụng file CSV mặc định: {csv_path}")
    
    # Load dữ liệu và khởi tạo Mini-RAG System
    print("[*] Đang load dữ liệu...")
    df, rag_system = load_and_analyze_data(csv_path)
    
    # Hiển thị thông tin model cố định
    print("=" * 60)
    print("[AUTOMOBILE DATA - MINI-RAG CHART GENERATOR]")
    print("=" * 60)
    print(f"Model đang sử dụng: {SELECTED_MODEL}")
    print("9Router sẽ tự fallback theo combo model đã cấu hình trên local web.")
    print("=" * 60 + "\n")
    
    # Nhập user prompt từ console
    while True:
        user_prompt = input("Nhập yêu cầu vẽ biểu đồ ('q' để quit): ").strip()
        
        # Kiểm tra quit
        if user_prompt.lower() == 'q':
            print("[*] Thoát chương trình.")
            sys.exit(0)
        
        # Kiểm tra request có hợp lệ không
        if not is_valid_chart_request(user_prompt):
            print("[✗] Yêu cầu không hợp lệ, vui lòng nhập lại yêu cầu về vẽ biểu đồ.\n")
            continue
        
        print(f"\nYêu cầu: {user_prompt}")
        print(f"Model đang sử dụng: {SELECTED_MODEL}\n")
        
        # Gọi hàm generate_and_run_chart với mini-RAG system
        generate_and_run_chart(user_prompt, df, rag_system)
        print("\n" + "=" * 60 + "\n")
