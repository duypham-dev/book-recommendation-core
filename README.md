# Hướng Dẫn Triển Khai Hệ Thống Gợi Ý Sách

Tài liệu này hướng dẫn chi tiết cách huấn luyện và triển khai hai mô hình gợi ý:
- **HybridImplicitSBERT**: Kết hợp Implicit ALS + SBERT
- **HybridNeural**: Kết hợp Neural Collaborative Filtering (NCF) + SBERT


## 📦 Cài Đặt Môi Trường

### 1. Clone Repository

```bash
git clone https://github.com/DUYBEGINER/Book_recommendation_system_core.git
cd Book_recommendation_system_core/RS
```

### 2. Tạo Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies chính:**
- `implicit`: Collaborative filtering (ALS)
- `torch`: PyTorch cho NCF
- `sentence-transformers`: SBERT embeddings
- `fastapi`, `uvicorn`: Web API
- `scikit-learn`, `pandas`, `numpy`: Data processing
- `psycopg2-binary`, `sqlalchemy`: Database

---

##  Cấu Hình Database

### 1. Tạo File `.env`

Tạo file `.env` trong thư mục `RS/`:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=book_recommendation_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_SCHEMA=book_recommendation_system

# Model Configuration
ALPHA=0.6
ARTIFACTS_DIR=./artifacts
```

### 2. Cấu Trúc Database

Hệ thống yêu cầu các bảng sau trong schema `book_recommendation_system`:

**Bảng `books`, `authors`, `book_authors`, `genres`, `book_genres`:**
- Được nêu trong hướng dẫn source backend

### 3. Kiểm Tra Kết Nối

```bash
python -c "from src.data.db_loader import DatabaseLoader; from src.utils.config import get_settings; s = get_settings(); loader = DatabaseLoader(s.db_uri, s.db_schema); print(f'Books: {len(loader.load_books())}'); print('Database connected!')"
```

---

## Huấn Luyện Mô Hình

### Model 1: HybridImplicitSBERT (ALS + SBERT)

#### Huấn Luyện Cơ Bản

```bash
python train.py
```

#### Huấn Luyện Với Đánh Giá

```bash
python train.py --evaluate --test-ratio 0.2
```

#### Tùy Chỉnh Tham Số

```bash
python train.py \
  --alpha 0.4 \
  --als-factors 64 \
  --als-iterations 30 \
  --als-regularization 0.01 \
  --artifacts-dir ./artifacts \
  --evaluate
```

**Tham số quan trọng:**

| Tham Số | Mô Tả | Giá Trị Mặc Định | Khuyến Nghị |
|---------|-------|------------------|-------------|
| `--alpha` | Trọng số ALS (0-1), SBERT = 1-alpha | 0.4 | 0.3-0.5 |
| `--als-factors` | Số chiều latent factors | 64 | 32-128 |
| `--als-iterations` | Số vòng lặp ALS | 30 | 20-50 |
| `--als-regularization` | Hệ số regularization | 0.01 | 0.001-0.1 |
| `--device` | Device (cuda/cpu) | auto | cuda nếu có GPU |
---

### Model 2: HybridNeural (NCF + SBERT)

#### Huấn Luyện Cơ Bản

```bash
python train_neural.py
```

#### Huấn Luyện Với Đánh Giá

```bash
python train_neural.py --evaluate --test-ratio 0.2
```

#### Tùy Chỉnh Tham Số

```bash
python train_neural.py \
  --alpha 0.6 \
  --gmf-dim 64 \
  --ncf-epochs 20 \
  --ncf-batch-size 256 \
  --device cuda \
  --artifacts-dir ./artifacts_neural \
  --evaluate
```

**Tham số quan trọng:**

| Tham Số | Mô Tả | Giá Trị Mặc Định | Khuyến Nghị |
|---------|-------|------------------|-------------|
| `--alpha` | Trọng số NCF (0-1), SBERT = 1-alpha | 0.6 | 0.5-0.7 |
| `--gmf-dim` | Số chiều GMF embedding | 64 | 32-128 |
| `--ncf-epochs` | Số epochs huấn luyện NCF | 20 | 10-30 |
| `--ncf-batch-size` | Batch size cho NCF | 256 | 128-512 |
| `--device` | Device (cuda/cpu) | auto | cuda (bắt buộc GPU) |


---

## 🚀 Khởi Động Servers

### Server 1: HybridImplicitSBERT (Port 8003)

**Khởi động:**
```bash
python server.py
```

**Output (ví dụ):**
```
🚀 Starting Hybrid Implicit ALS + SBERT Recommender Server...
Loading Implicit ALS + SBERT models from ./artifacts...
✅ Models loaded successfully!
  ALS users: 1000
  ALS items: 4500
  SBERT books: 5000
  SBERT profiles: 1000
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

### Server 2: HybridNeural (Port 8002)

**Khởi động:**
```bash
python server_neural.py
```

**Output:**
```
🚀 Starting Hybrid Neural Recommender Server...
Loading neural models from ./artifacts_neural...
✅ Neural models loaded successfully!
  NCF users: 1000
  NCF items: 4500
  SBERT books: 5000
  SBERT profiles: 1000
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
```

### Chạy Đồng Thời Nhiều Servers

**Windows (PowerShell):**
```powershell
# Terminal 1
python server.py

# Terminal 2 (mở terminal mới)
python server_neural.py
```

## 🧪 Kiểm Tra API

### Health Check

**HybridImplicitSBERT (Port 8003):**
```bash
curl http://localhost:8003/api/v1/health
```

**HybridNeural (Port 8002):**
```bash
curl http://localhost:8002/api/v1/health
```

**Response mẫu:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "als_users": 1000,
  "als_items": 4500,
  "sbert_books": 5000,
  "sbert_profiles": 1000
}
```

### Gợi Ý ClicitSBERT(ví dụ)

```bash
# HybridImplicitSBERT
curl "http://localhost:8003/api/v1/recommendations?user_id=123&limit=10"

# HybridNeural
curl "http://localhost:8002/api/v1/recommendations?user_id=123&limit=10"
```

**Response mẫu:**
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "book_id": 456,
      "score": 0.8765,
      "reasons": {
        "als": 0.85,
        "sbert": 0.65,
        "pop": 0.0
      }
    },
    ...
  ],
  "count": 10
}
```

### Sách Tương Tự (SBERT)

```bash
curl "http://localhost:8003/api/v1/similar?book_id=456&limit=5"
```

**Response:**
```json
{
  "book_id": 456,
  "similar_books": [
    {
      "book_id": 789,
      "score": 0.9234,
      "source": "sbert_similarity"
    },
    ...
  ]
}
```

### Gợi Ý Đa Dạng (Chỉ HybridImplicitSBERT)

```bash
curl "http://localhost:8003/api/v1/diversity?book_id=456&limit=5"
```

**Response:**
```json
{
  "book_id": 456,
  "items": [
    {
      "book_id": 789,
      "rating": 4.5,
      "score": 0.8523,
      "metadata": {
        "genre_diversity": 0.85,
        "author_diversity": 0.72
      }
    },
    ...
  ]
}
```

### Ghi Nhận Feedback (Online Learning - Chỉ ImplicitSBERT)

```bash
curl -X POST "http://localhost:8003/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "book_id": 456,
    "event": "rating",
    "rating_value": 5
  }'
```

**Các loại event:**
- `rating`: Đánh giá (rating_value: 1-5)
- `favorite`: Yêu thích (rating_value: 0 để bỏ yêu thích)
- `history`: Lịch sử đọc

### Trạng Thái Online Learning

```bash
curl "http://localhost:8003/api/v1/online-learning/status"
```

**Response:**
```json
{
  "enabled": true,
  "buffer_size": 45,
  "buffer_capacity": 100,
  "buffer_full": false,
  "note": "Only SBERT profiles updated incrementally. ALS requires full retrain."
}
```

### Trigger Incremental Update (Manual)

```bash
curl -X POST "http://localhost:8003/api/v1/online-learning/update?force=true"
```

---

## 🎯 Tham Số Tối Ưu

### Lựa Chọn Mô Hình

**Chọn HybridImplicitSBERT khi:**
- Hệ thống yêu cầu phản hồi nhanh (<100ms)
- Cần online learning để cập nhật real-time
- Tài nguyên hạn chế (không có GPU)


**Chọn HybridNeural khi:**
- Ưu tiên độ chính xác cao nhất
- Có GPU mạnh cho training
- Có thể retrain định kỳ (không cần online learning)

### Điều Chỉnh Alpha

**Alpha** quyết định tỷ trọng giữa Collaborative (ALS/NCF) và Content-based (SBERT):

```python
final_score = alpha * CF_score + (1 - alpha) * SBERT_score
```

| Alpha | Ý Nghĩa | Khi Nào Dùng |
|-------|---------|--------------|
| **0.3-0.4** | Ưu tiên SBERT | Nhiều sách mới, cold-start cao |
| **0.5** | Cân bằng | Dữ liệu đa dạng |
| **0.6-0.7** | Ưu tiên CF | Nhiều tương tác, ít cold-start |

**Thử nghiệm:**
```bash
# Test với alpha thấp (ưu tiên content)
python train.py --alpha 0.3 --evaluate

# Test với alpha cao (ưu tiên collaborative)
python train.py --alpha 0.7 --evaluate
```

**Cập nhật:** 2025-11-05
