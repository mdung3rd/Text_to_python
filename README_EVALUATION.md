# README_EVALUATION

Hướng dẫn ngắn gọn cách evaluate model sinh code vẽ biểu đồ cho dataset Automobile.

Project hỗ trợ 2 kiểu evaluate:

1. **Local LLM**: dùng Ollama local.
2. **Cloud API**: dùng OpenAI-compatible API, ví dụ Groq.

---

## 1. Các file chính

| File | Mục đích |
|---|---|
| `100_queries_from_automobile.csv` | Bộ 100 câu hỏi dùng để evaluate. |
| `evaluate_model_robust.py` | Evaluate Local LLM/Ollama. |
| `evaluate_model_cloudapi.py` | Evaluate Cloud API. |
| `run_evaluation.py` | Menu chọn evaluate Local hoặc Cloud. |
| `local_llm.py` | Chạy thử Local LLM dạng nhập prompt thủ công. |
| `cloud_llm.py` | Chạy thử Cloud/Groq dạng nhập prompt thủ công. |
| `query.py` | Sinh file queries test từ dataset Automobile. |

---

## 2. Cài thư viện

```powershell
pip install pandas requests matplotlib seaborn numpy openai
```

hoặc:

```powershell
py -m pip install pandas requests matplotlib seaborn numpy openai
```

---

## 3. Kiểm tra đường dẫn dữ liệu

Các file evaluation đang dùng dataset:

```python
C:\Users\DELL\Downloads\archive\Automobile_data.csv
```

Nếu máy khác không có đường dẫn này, sửa lại trong:

- `evaluate_model_robust.py`
- `evaluate_model_cloudapi.py`
- `cloud_llm.py`
- `local_llm.py`
- `query.py`

File queries dùng để evaluate:

```text
100_queries_from_automobile.csv
```

Format chính:

```csv
Query,Expected_Chart_Type,Required_Columns,Difficulty
```

---

## 4. Evaluate Local LLM bằng Ollama

File dùng:

```text
evaluate_model_robust.py
```

Model mặc định:

```python
OLLAMA_MODEL = "llama3.2-lite"
```

### Bước 1: chạy Ollama

```powershell
ollama serve
```

Nếu chưa có model:

```powershell
ollama pull llama3.2-lite
```

### Bước 2: chạy evaluation

```powershell
py evaluate_model_robust.py
```

hoặc:

```powershell
python evaluate_model_robust.py
```

Kết quả lưu trong:

```text
test_results/
```

Tên file:

```text
evaluation_results_robust_*.csv
evaluation_summary_robust_*.txt
```

---

## 5. Evaluate Cloud API

File dùng:

```text
evaluate_model_cloudapi.py
```

Nếu dùng Groq, cấu hình client dạng:

```python
from openai import OpenAI

GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

GROQ_MODEL = "llama-3.1-8b-instant"
```

Khi gọi API dùng:

```python
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
)
```

Chạy evaluation:

```powershell
py evaluate_model_cloudapi.py
```

hoặc:

```powershell
python evaluate_model_cloudapi.py
```

Kết quả lưu trong:

```text
test_results/
```

Tên file:

```text
evaluation_results_cloudapi_*.csv
evaluation_summary_cloudapi_*.txt
```

---

## 6. Chạy bằng menu

Có thể dùng `run_evaluation.py` để chọn mode:

```powershell
py run_evaluation.py
```

Menu chính:

```text
1. Ollama Local
2. Cloud API
3. Compare both results
4. Exit
```

Nên chọn:

```text
3. Run and show results
```

để vừa chạy evaluation vừa xem kết quả tóm tắt.

Lưu ý: chức năng compare trong menu hiện mới báo `Coming soon`, chưa compare tự động.

---

## 7. Metrics được tính

Evaluation tính các chỉ số chính:

| Metric | Ý nghĩa |
|---|---|
| `Runnable Rate` | Tỷ lệ code model sinh ra chạy được. |
| `Chart-Type Accuracy` | Tỷ lệ model sinh đúng loại biểu đồ mong đợi. |
| `Conditional Accuracy` | Accuracy tính trên các code chạy được. |
| `By Chart Type` | Thống kê theo từng loại chart: Bar, Scatter, Histogram, Boxplot,... |
| `By Difficulty` | Thống kê theo độ khó: Easy, Medium, Hard. |

Công thức:

```text
Runnable Rate = runnable / total * 100
Chart-Type Accuracy = correct chart / total * 100
Conditional Accuracy = correct chart among runnable / runnable * 100
```

---

## 8. File kết quả CSV

Output CSV có các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `query_id` | ID query. |
| `query` | Nội dung yêu cầu. |
| `expected_chart` | Loại chart mong đợi. |
| `detected_chart` | Loại chart detect từ code sinh ra. |
| `chart_match` | Đúng chart hay không. |
| `runnable` | Code chạy được hay không. |
| `error` | Lỗi nếu code không chạy được. |
| `api_time` | Thời gian gọi model/API. |
| `code_length` | Độ dài code sinh ra. |
| `difficulty` | Độ khó query. |

---

## 9. Test nhanh ít queries

Mặc định script chạy 100 queries:

```python
for idx, row in queries_df.iterrows():
```

Muốn test nhanh 10 câu thì sửa thành:

```python
for idx, row in queries_df.head(10).iterrows():
```

Có thể sửa trong:

- `evaluate_model_robust.py`
- `evaluate_model_cloudapi.py`

---

## 10. Lỗi thường gặp

### Không tìm thấy file CSV

Kiểm tra lại các biến path:

```python
CSV_QUERIES_PATH
CSV_AUTOMOBILE_PATH
OUTPUT_DIR
SCRIPT_DIR
```

### Không kết nối được Ollama

Chạy:

```powershell
ollama serve
```

Kiểm tra:

```powershell
curl http://127.0.0.1:11434/api/tags
```

### Cloud API lỗi key/model

Kiểm tra:

- API key đúng chưa.
- `base_url` đúng chưa.
- Tên model đúng chưa.

Với Groq:

```python
base_url="https://api.groq.com/openai/v1"
```

### GitHub chặn push vì API key

Không nên push API key thật. Đổi key thành placeholder trước khi commit:

```python
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
```

---

## 11. Quy trình evaluate đề xuất

### Local LLM

```powershell
ollama serve
py evaluate_model_robust.py
```

### Cloud API

```powershell
py evaluate_model_cloudapi.py
```

### Menu

```powershell
py run_evaluation.py
```

Sau khi chạy xong, xem file kết quả trong:

```text
test_results/
```
