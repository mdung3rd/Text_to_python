# 🧪 TEST & EVALUATE MODEL - COMPLETE SETUP

## 📦 Bạn vừa nhận được gì:

### 3 Files chính:

1. **`evaluate_model_robust.py`** (150+ lines)
   - Main evaluation script
   - Test 100 queries từ CSV
   - Tính toán metrics
   - Lưu kết quả chi tiết

2. **`run_evaluation.py`** (150+ lines)
   - Interactive menu
   - Dễ sử dụng
   - Hiển thị kết quả ngay

3. **`EVALUATION_GUIDE.md`**
   - Hướng dẫn chi tiết
   - Cách hiểu kết quả
   - Tips & tricks

---

## 🚀 QUICK START (3 bước):

### Bước 1: Khởi động Ollama
```bash
# Terminal 1 - Mở cmd hoặc PowerShell
ollama serve
# Chờ tới khi thấy "listening on"
```

### Bước 2: Chạy evaluation
```bash
# Terminal 2 - Chạy script
cd C:\vscode\python
python run_evaluation.py

# Chọn option 1 hoặc 3 để chạy test
```

### Bước 3: Xem kết quả
- Kết quả sẽ hiển thị trong console
- Files được lưu tại: `C:\vscode\python\test_results\`
  - `evaluation_results_*.csv` - Chi tiết mỗi query
  - `evaluation_summary_*.txt` - Tóm tắt

---

## 📊 METRICS ĐƯỢC TÍNH:

### 🎯 **Runnable Rate** (%)
- Tỷ lệ % code sinh ra chạy được (không error)
- Kiểm tra: Syntax errors, runtime errors
- **Công thức**: `(Runnable count / Total) × 100%`

### 📈 **Chart-Type Accuracy** (%)
- Tỷ lệ % dự đoán đúng loại chart
- So sánh: Code sinh ra vs Expected_Chart_Type trong CSV
- **Công thức**: `(Correct count / Total) × 100%`

### 🔀 **Chi tiết theo Chart Type**
- Runnable rate cho mỗi loại: Bar, Scatter, Histogram, Boxplot, Pie, etc.
- Accuracy cho mỗi loại

### 🏆 **Chi tiết theo Difficulty**
- Easy, Medium, Hard
- Metrics riêng cho mỗi mức độ

---

## 📋 CSV OUTPUT FORMAT:

| Cột | Ý nghĩa |
|-----|---------|
| query_id | ID (1-100) |
| query | Nội dung query |
| expected_chart | Chart type mong đợi (Bar, Scatter, ...) |
| detected_chart | Chart type phát hiện từ code |
| chart_match | Có khớp không (Yes/No) |
| runnable | Code chạy được không (Yes/No) |
| error | Chi tiết lỗi (nếu có) |
| api_time | Thời gian API call (seconds) |
| code_length | Số ký tự code sinh ra |
| difficulty | Mức độ khó (Easy/Medium/Hard) |

**Ví dụ:**
```
query_id,query,expected_chart,detected_chart,chart_match,runnable,...
1,"Plot the mean width grouped by aspiration.",Bar,Bar,Yes,Yes,...
2,"Draw a scatter plot comparing city-mpg and price.",Scatter,Scatter,Yes,Yes,...
3,"Show a histogram of price.",Histogram,Bar,No,Yes,...
```

---

## 🔧 CUSTOMIZE OPTIONS:

### 1. Test ít queries hơn (để debug nhanh):
```python
# Trong evaluate_model.py, thay dòng:
for idx, row in queries_df.iterrows():

# Thành:
for idx, row in queries_df.head(10).iterrows():  # Chỉ 10 queries
```

### 2. Đổi model Ollama:
```python
OLLAMA_MODEL = "mistral"  # thay vì "llama3.2-lite"
OLLAMA_MODEL = "neural-chat"
```

### 3. Tăng/giảm timeout:
```python
OLLAMA_TIMEOUT = 180  # từ 120 lên 180 seconds
```

### 4. Điều chỉnh temperature (control randomness):
```python
OLLAMA_TEMPERATURE = 0.1  # từ 0.2 xuống 0.1 (ít random)
```

---

## 💡 HIỂU KẾT QUẢ:

### Ví dụ output:
```
📈 OVERALL METRICS:
   Total queries:          100
   Runnable:               85/100 (85.0%)          ← 85% code chạy được
   Chart type correct:     72/100 (72.0%)          ← 72% predict đúng
   Accuracy (runnable):    72/85 (84.7%)           ← 84.7% đúng trong những cái chạy được

📊 BY CHART TYPE:
   Bar          | Count: 35 | Runnable: 32 (91.4%) | Correct: 28 (80.0%)
   Scatter      | Count: 15 | Runnable: 13 (86.7%) | Correct: 11 (73.3%)
   Histogram    | Count: 20 | Runnable: 18 (90.0%) | Correct: 15 (75.0%)
```

### Giải thích:
- **Bar chart**: 35 queries
  - 32 chạy được (91.4% runnable)
  - 28 predict đúng (80% accuracy)
  - → Model yếu nhất ở accuracy, mạnh ở runnable

---

## 🐛 TROUBLESHOOTING:

### ❌ "Cannot connect to Ollama"
**Giải pháp:**
```bash
# Kiểm tra Ollama đang chạy:
curl http://127.0.0.1:11434/api/tags

# Nếu không, khởi động:
ollama serve

# Nếu model không có:
ollama pull llama3.2-lite
```

### ❌ "SyntaxError in generated code"
**Nguyên nhân:** Model sinh code chứa lỗi syntax
**Giải pháp:**
1. Giảm temperature: `0.1` (ít random)
2. Thêm ví dụ code tốt trong prompt
3. Dùng model khác

### ❌ Script quá chậm
**Giải pháp:**
1. Giảm `OLLAMA_MAX_TOKENS`: 384 → 256
2. Giảm `OLLAMA_TEMPERATURE`: 0.2 → 0.1 (call lại nhanh hơn)
3. Test 10 queries trước: `queries_df.head(10)`

### ❌ "ModuleNotFoundError"
**Giải pháp:**
```bash
# Cài đặt thư viện cần thiết:
pip install pandas requests matplotlib seaborn numpy
```

---

## 📊 ANALYZE RESULTS:

Sau khi có kết quả CSV, bạn có thể:

### 1. Mở trong Excel/Python:
```python
import pandas as pd

df = pd.read_csv("C:\\vscode\\python\\test_results\\evaluation_results_*.csv")

# Lọc những cái lỗi
failed = df[df['runnable'] == 'No']
print(failed[['query', 'error']])

# Lọc predict sai chart
wrong_chart = df[(df['chart_match'] == 'No') & (df['runnable'] == 'Yes')]
print(wrong_chart[['query', 'expected_chart', 'detected_chart']])
```

### 2. Tính toán riêng:
```python
# Runnable rate by difficulty
easy = df[df['difficulty'] == 'Easy']
print(f"Easy runnable: {(easy['runnable'] == 'Yes').sum() / len(easy)}")
```

---

## 🎯 CẢI THIỆN MODEL:

Dựa vào kết quả, bạn có thể:

### Nếu Runnable Rate thấp (< 70%):
1. **Cải tiến prompt** - yêu cầu code chính xác hơn
2. **Thêm ví dụ** - in-context learning
3. **Giảm temperature** - ít lỗi
4. **Đổi model** - thử model khác

### Nếu Chart-Type Accuracy thấp (< 70%):
1. **Explicit instruction** - "Generate a BAR chart using plt.bar()"
2. **Prompt engineering** - yêu cầu rõ ràng loại chart
3. **Few-shot learning** - thêm ví dụ code cho mỗi chart type

### Nếu một chart type yếu (ví dụ Pie < 50%):
1. Thêm ví dụ pie chart đặc biệt trong prompt
2. Kiểm tra data xem có phù hợp với pie không
3. Improve prompt cho pie charts

---

## 📝 NEXT STEPS:

1. **Chạy evaluation**: `python run_evaluation.py`
2. **Xem kết quả**: CSV output
3. **Phân tích**: Identify yếu điểm
4. **Cải thiện**: Fix prompt/model
5. **Re-test**: Chạy lại để so sánh

---

## 📚 FILES REFERENCE:

```
C:\vscode\python\
├── 100_queries_from_automobile.csv       ← Input: 100 queries
├── test_api_local_ollama.py              ← Original test script
├── evaluate_model.py                     ← Main evaluation script
├── run_evaluation.py                     ← Interactive menu
├── EVALUATION_GUIDE.md                   ← Detailed guide
├── README_EVALUATION.md                  ← This file
└── test_results/
    ├── evaluation_results_YYYYMMDD_HHMMSS.csv
    └── evaluation_summary_YYYYMMDD_HHMMSS.txt
```

---

## 🎓 KEY FORMULAS:

### Runnable Rate:
```
Runnable Rate (%) = (# queries where code runs without error / total queries) × 100
```

### Chart-Type Accuracy:
```
Chart Accuracy (%) = (# queries where detected chart matches expected / total queries) × 100
```

### Conditional Accuracy (accuracy among runnable):
```
Conditional Accuracy (%) = (# runnable queries with correct chart / # runnable queries) × 100
```

---

**Ready? Let's go! 🚀**

```bash
python C:\vscode\python\run_evaluation.py
```

---

*Created: 2024*
*Purpose: Evaluate LLM code generation and chart type accuracy*
