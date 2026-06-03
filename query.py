import pandas as pd
import random
import itertools

# Chốt seed để kết quả 100 câu luôn cố định mỗi lần chạy
random.seed(42)

# 1. Đọc trực tiếp từ file automobile.csv của Dũng đang có
try:
    df_auto = pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\Automobile_data.csv", na_values=['?'], encoding="ISO-8859-1")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file automobile.csv. Nhớ để cùng thư mục với code nhé!")
    exit()

# 2. Tự động nhận diện cột số (để vẽ Scatter, Histogram) và cột chữ (để vẽ Bar, Pie)
num_cols = df_auto.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_auto.select_dtypes(include=['object', 'category']).columns.tolist()

# Xóa bớt các cột không mang nhiều ý nghĩa thống kê (nếu có)
if 'symboling' in num_cols: num_cols.remove('symboling')

test_cases = []

# Mẫu câu truy vấn đa dạng
bar_phrases = ["Plot a bar chart showing the count of each {}.", "Visualize the frequency of {} using a bar plot."]
hist_phrases = ["Visualize the distribution of {} using a histogram.", "Plot a histogram for {}."]
scatter_phrases = ["Draw a scatter plot comparing {} and {}.", "Show the relationship between {} and {}."]
mean_phrases = ["Show a bar chart of the average {} for each {}.", "Plot the mean {} grouped by {}."]
box_phrases = ["Show a boxplot of {} grouped by {} to find outliers."]

# 3. Sinh dữ liệu dựa trên các cột thực tế vừa đọc được
# Nhóm Bar Chart đếm số lượng (Easy)
for col in cat_cols[:10]: # Lấy 10 cột chữ đầu tiên
    test_cases.append({"Query": random.choice(bar_phrases).format(col), "Expected_Chart_Type": "Bar", "Required_Columns": col, "Difficulty": "Easy"})

# Nhóm Histogram (Easy)
for col in num_cols:
    test_cases.append({"Query": random.choice(hist_phrases).format(col), "Expected_Chart_Type": "Histogram", "Required_Columns": col, "Difficulty": "Easy"})

# Nhóm Scatter Plot tương quan (Medium)
scatter_pairs = list(itertools.combinations(num_cols, 2))
random.shuffle(scatter_pairs)
for col1, col2 in scatter_pairs[:30]: 
    test_cases.append({"Query": random.choice(scatter_phrases).format(col1, col2), "Expected_Chart_Type": "Scatter", "Required_Columns": f"{col1}, {col2}", "Difficulty": "Medium"})

# Nhóm tính Trung bình (Medium) & Boxplot (Hard)
for _ in range(30):
    c_col = random.choice(cat_cols)
    n_col = random.choice(num_cols)
    test_cases.append({"Query": random.choice(mean_phrases).format(n_col, c_col), "Expected_Chart_Type": "Bar", "Required_Columns": f"{c_col}, {n_col}", "Difficulty": "Medium"})
    test_cases.append({"Query": random.choice(box_phrases).format(n_col, c_col), "Expected_Chart_Type": "Boxplot", "Required_Columns": f"{c_col}, {n_col}", "Difficulty": "Hard"})

# 4. Trộn ngẫu nhiên, lọc trùng và chốt đúng 100 câu
df_test = pd.DataFrame(test_cases)
df_test = df_test.drop_duplicates(subset=['Query']).sample(n=100, random_state=42, replace=True).reset_index(drop=True)

# 5. Xuất ra file để đưa cho Tân
df_test.to_csv('100_queries_from_real_data.csv', index=False, encoding='utf-8')
print("Thành công! Đã quét file automobile.csv và tạo ra 100_queries_from_real_data.csv")