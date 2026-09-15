# Cài đặt thư viện

```bash
pip install -r requirements.txt
```

| Thư viện | Dùng cho |
|---|---|
| `pycurl` | `Crawl Code.py` — đo DNS/TCP/TLS/response time qua HTTP request |
| `pythonping` | `Crawl Code.py` — đo RTT (ping) |
| `numpy` | `main.py` |
| `pandas` | `main.py`, `Crawl Code.py` |
| `scikit-learn` | `main.py` — train/test split, Isolation Forest, scaler, 3 model ML, metrics |
| `matplotlib` | `main.py` — toàn bộ biểu đồ |

`io`, `urllib.parse`, `datetime`, `pathlib` là thư viện chuẩn của Python (stdlib), không cần cài thêm.

**Lưu ý:** `pythonping` cần quyền root/admin để mở raw socket ping. Chạy `Crawl Code.py` bằng `sudo` (Linux/Mac) hoặc terminal admin (Windows), nếu không bước ping sẽ lỗi.

---

# Thứ tự chạy

### 1. (Tùy chọn) Thu thập dữ liệu mới — `Crawl Code.py`
```bash
sudo python3 "Crawl Code.py"
```
Ghi dữ liệu vào `inputs/Dataset_finale.csv` (append nếu file đã tồn tại). Bỏ qua bước này nếu dùng luôn `Dataset_finale.csv` có sẵn trong `inputs/`.

### 2. Chạy pipeline — `main.py`
```bash
python3 main.py
```
Đọc `inputs/Dataset_finale.csv`, chạy tuần tự 7 bước, ghi toàn bộ kết quả (CSV + PNG) vào `outputs/`:

```
1. load_data          -> đọc CSV
2. clean               -> null/dup, http_status, cross-field check, split train/test, Isolation Forest (fit train)
3. analyze              -> describe() + domain_category counts (train)
4. feature_engineer      -> cyclical time encoding, one-hot domain_category, StandardScaler (fit train)
   [Pearson correlation chạy song song, ngoài pipeline, ngay sau bước 4]
5. train_models            -> baseline_mean, linear_regression, decision_tree, random_forest
6. evaluate                  -> MAE/RMSE/R²/MAPE/Accuracy + coefficients/importances + top-10 lỗi dự đoán (3 model non-baseline)
7. visualize                   -> feature_vs_target scatter + actual_vs_predicted/residual_plot/residual_hist mỗi model
```

Không cần chỉnh sửa gì thủ công — `outputs/` tự tạo nếu chưa có.
