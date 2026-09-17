from sklearn.datasets import make_moons
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 1. Tạo dữ liệu giả lập
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# 2. Định nghĩa cấu hình K-Fold (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Khai báo lưới tham số cần tìm kiếm
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20]
}

# 4. Khởi tạo GridSearchCV kết hợp K-Fold
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=kf,              # Sử dụng K-Fold đã định nghĩa
    scoring='accuracy', # Tiêu chí đánh giá
    n_jobs=-1           # Chạy song song trên toàn bộ nhân CPU
)

# 5. Chạy tìm kiếm siêu tham số
grid_search.fit(X, y)

# 6. Hiển thị kết quả tối ưu
print(f"Tham số tối ưu: {grid_search.best_params_}")
print(f"Độ chính xác K-Fold cao nhất: {grid_search.best_score_:.4f}")

# 7. Trích xuất mô hình hoàn chỉnh đã tối ưu
best_model = grid_search.best_estimator_