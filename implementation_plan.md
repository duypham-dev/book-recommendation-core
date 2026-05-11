# Đánh Giá & Kế Hoạch Tối Ưu: Book Recommendation System

## Tóm Tắt Kiến Trúc Hiện Tại

Hệ thống đang dùng kiến trúc **Hybrid ALS + SBERT**:
- **ALS (Implicit)**: Collaborative Filtering cho implicit feedback
- **SBERT** (`keepitreal/vietnamese-sbert`): Content-Based Filtering với dense semantic embeddings
- **Online Learning**: Incremental update SBERT user profiles (ALS không được cập nhật)
- **Callback**: Thông báo Java backend để invalidate cache sau retrain

---

## Phân Tích: Điểm Đã Tốt ✅

| Hạng mục | Nhận xét |
|---|---|
| Hybrid architecture | Kết hợp ALS + SBERT là hướng đúng, bù trừ cold-start |
| Temporal split | `DataSplitter.temporal_split` chia data đúng kiểu time-based (không dùng random) |
| Implicit signal | Dùng `implicit.als` đúng với BPR/ALS cho feedback ngầm định |
| Popularity fallback | Có fallback cho cold-start user, đúng thiết kế |
| Callback mechanism | Tích hợp HTTP callback về Java backend để invalidate cache |
| L2 Normalization | Embeddings và user profiles đều được L2 normalize trước khi tính cosine |
| Score fusion | Min-max normalize trước khi combine ALS + SBERT scores |
| Diversity endpoint | Có riêng endpoint `/diversity` cho book-to-book recommendations |

---

## Phân Tích: Vấn Đề & Thiếu Sót 🔴

### 1. Data Loading — `db_loader.py`

**Vấn đề nghiêm trọng:**

```python
# Vấn đề: Kết hợp 3 loại signal với thang điểm không nhất quán
rating_value::float AS strength       -- scale: 1-5
GREATEST(1.0, COALESCE(progress/100.0, 0.5) * 5.0)  -- scale: 1.0–5.0 nhưng logic tối
5.0 AS strength  -- Favorites = max rating (hard-coded)
```

**Cụ thể:**
- `reading_history`: `GREATEST(1.0, progress/100.0 * 5)` khi progress = 20% → strength = 1.0. Nhưng khi progress = 0 → `GREATEST(1.0, 0.5*5) = 2.5`. **Logic lộn xộn**: progress = 0 lại mạnh hơn progress = 20%.
- **Không có deduplication logic**: Nếu một user đọc sách 3 lần, sẽ có 3 dòng trong `reading_history` → ALS bị tính cùng 1 interaction 3 lần trước khi aggregate.
- **Không có timestamp filtering**: Load toàn bộ data, kể cả interaction cũ từ nhiều năm trước. Không có time decay.
- **`print()` statement còn sót lại** (debug code) ở `db_loader.py:36` và `collaborative.py:33`.

---

### 2. Interaction Strength Mapping — `routes.py`

```python
elif request.event == 'history':
    strength = 1.0  # Simple implicit signal: user read the book
```

**Vấn đề**: Online feedback `history` luôn = 1.0, trong khi batch training có dùng `progress`-based strength. Hai pipeline **không đồng nhất**. Model được train với 1 phân phối, nhưng được update với phân phối khác.

---

### 3. Retrain — `routes.py` (hàm `retrain_models`)

```python
async def retrain_models():
    global is_retraining, recommender
    ...
    recommender.train(books_df, interactions_df)  # Mutable global state!
    recommender.save(artifacts_dir)
```

**Vấn đề nghiêm trọng:**
- **Race condition**: `recommender` là global variable, retrain đang diễn ra trong background nhưng các request khác vẫn đang đọc `recommender`. Khi `recommender.train()` được gọi, internal state của model bị overwrite → request đang serve có thể đọc dữ liệu bị corrupt.
- **`is_retraining` không thread-safe**: Được đọc/ghi từ coroutine (async) và background task (sync), không dùng `asyncio.Lock` hay `threading.Lock`.
- **Không có model versioning**: Model mới thay thế ngay, không có rollback nếu retrain thất bại.
- **Retrain train lại từ đầu toàn bộ data**: Không tận dụng incremental update.

---

### 4. Online Learning — Tính Nhất Quán

```python
# ALS không được update trong online learning
# → Sau nhiều incremental update, SBERT và ALS càng ngày càng diverge
# Kết quả combine scores ngày càng kém ý nghĩa
```

**Vấn đề**: Qua thời gian, interaction buffer chỉ cập nhật SBERT profiles, không cập nhật ALS. Điều này dẫn đến:
- SBERT biết user mới thích thể loại gì
- ALS vẫn dùng data cũ → score ALS ngày càng stale
- Kết quả hybrid blend bị lệch, không đúng với thực tế

**Chưa có cơ chế tự động trigger retrain ALS** sau N ngày hay N interactions.

---

### 5. Interaction Deduplication & Aggregation

```python
# collaborative.py
agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
```

Dùng `sum` để aggregate là **đúng cho ALS implicit** (tổng confidence), nhưng:
- Không có **upper cap** → Một user đọc 1 cuốn sách 100 lần sẽ có strength = 500, dominate toàn bộ signal.
- Không có **normalization per user** → User hoạt động nhiều sẽ chiếm ưu thế trong training.

---

### 6. Evaluation — `evaluation.py`

```python
def _user_in_model(recommender, user_id):
    if hasattr(recommender, 'cf_model') ...  # Attribute không tồn tại trong HybridRecommender
```

- **Bug**: `HybridRecommender` dùng `als_model`, nhưng evaluator check `cf_model` và `ncf_model` → cold-start check **luôn return False**, có nghĩa là tất cả user đều bị mark là cold-start và bị skip!
- **Leakage risk trong temporal split**: Split theo `test_ratio` = tỷ lệ cuối của toàn bộ data, nhưng không đảm bảo mỗi user có ít nhất 1 interaction trong test set.

---

### 7. Config — `config.py`

```python
@property
def db_uri(self) -> str:
    return f"postgresql://{self.db_user}:{self.db_password}@localhost:{self.db_port}/{self.db_name}"
```

**Vấn đề**: `localhost` bị hard-code trong `db_uri`. Khi chạy trong Docker container, phải dùng hostname của service (`postgres`), không phải `localhost`. Biến `db_host` đã có nhưng không được dùng.

---

### 8. Concurrency & State Management

- **`recommender` là module-level global**: Không an toàn khi dùng với multi-worker uvicorn (`--workers N`).
- **`interaction_buffer` là in-memory list**: Không persistent, restart server → mất toàn bộ buffer.
- **Online learning không thread-safe**: `self.interaction_buffer.append()` và `incremental_update()` có thể bị gọi đồng thời.

---

### 9. Artifacts Serialization

```python
pickle.dump({...}, f)  # Dùng pickle
```

- **Không có versioning** cho artifact format → Breaking change nếu thêm field mới.
- **pickle không an toàn** nếu artifact được load từ nguồn không tin cậy.
- **SBERT model weights không được lưu** (chỉ lưu embeddings), server khởi động lại phải tải lại từ HuggingFace → thêm dependency bên ngoài và latency khi startup.

---

## Kế Hoạch Tối Ưu — Ưu Tiên Cao 🔴

### Phase 1: Sửa Bug Nghiêm Trọng (1–2 ngày)

#### 1.1 Fix `db_uri` hard-coded `localhost`
**File**: `src/utils/config.py`

```python
# Trước
return f"postgresql://...@localhost:..."

# Sau
return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
```

#### 1.2 Fix evaluator `_user_in_model` — sai attribute name
**File**: `src/utils/evaluation.py`

```python
# Trước: check cf_model (không tồn tại)
if hasattr(recommender, 'cf_model') and recommender.cf_model is not None:

# Sau: check als_model (đúng attribute trong HybridRecommender)
if hasattr(recommender, 'als_model') and recommender.als_model is not None:
    if hasattr(recommender.als_model, 'user_id_map'):
        if user_id in recommender.als_model.user_id_map:
            return True
```

#### 1.3 Fix Race Condition trong Retrain — Atomic Swap
**File**: `src/api/routes.py`

```python
# Sau: train model mới vào instance riêng, swap atomically
async def retrain_models():
    global is_retraining, recommender
    try:
        is_retraining = True
        new_recommender = HybridRecommender(...)  # Instance mới
        new_recommender.train(books_df, interactions_df)
        new_recommender.save(artifacts_dir)
        recommender = new_recommender  # Atomic swap
    finally:
        is_retraining = False
```

#### 1.4 Remove debug `print()` statements
**Files**: `db_loader.py:36`, `collaborative.py:33`

---

### Phase 2: Cải Thiện Data Loading & Signal Quality (2–3 ngày)

#### 2.1 Fix interaction strength logic
**File**: `src/data/db_loader.py`

```sql
-- Trước: logic sai khi progress = 0 → strength = 2.5
GREATEST(1.0, COALESCE(progress/100.0, 0.5) * 5.0) AS strength

-- Sau: linear scale, progress = 0 → strength = 0.5 (minimal signal)
CASE
    WHEN progress IS NULL THEN 0.5
    ELSE GREATEST(0.5, (progress / 100.0) * 5.0)
END AS strength
```

#### 2.2 Thêm cap và time decay cho interactions

```sql
-- Chỉ lấy interactions trong 1 năm gần nhất (configurable)
-- Thêm time decay weight: interaction gần đây quan trọng hơn
SELECT user_id, book_id, 
    strength * EXP(-0.001 * EXTRACT(EPOCH FROM (NOW() - ts)) / 86400) AS strength
FROM ...
WHERE ts >= NOW() - INTERVAL '1 year'
```

#### 2.3 Strength cap per user-book pair

```python
# Sau aggregate, cap để tránh dominance
MAX_STRENGTH = 10.0
agg['strength'] = agg['strength'].clip(upper=MAX_STRENGTH)
```

#### 2.4 Đồng nhất strength mapping giữa online và batch

```python
# routes.py - feedback endpoint
elif request.event == 'history':
    progress = request.progress or 0  # Thêm field progress vào FeedbackRequest
    strength = max(0.5, (progress / 100.0) * 5.0)  # Đồng nhất với db_loader
```

---

### Phase 3: Retrain Lifecycle & Model Versioning (3–5 ngày)

#### 3.1 Model Versioning với timestamp

```
artifacts/
  v20260510_120000/      # Versioned model
    als_model.pkl
    sbert_model.pkl
    hybrid_recommender_metadata.pkl
  current -> v20260510_120000/   # Symlink đến version hiện tại
  previous -> v20260509_080000/  # Để rollback
```

```python
def save(self, artifacts_dir: Path):
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_dir = artifacts_dir / f'v{version}'
    versioned_dir.mkdir(parents=True, exist_ok=True)
    # ... save ...
    # Update symlink
    current_link = artifacts_dir / 'current'
    if current_link.exists(): current_link.unlink()
    current_link.symlink_to(versioned_dir.name)
```

#### 3.2 Tự động trigger retrain ALS

```python
# Thêm vào HybridRecommender
self.als_interaction_count = 0  # Count interactions kể từ lần retrain cuối
self.als_retrain_threshold = 1000  # Trigger ALS retrain sau N interactions

def add_interaction(self, ...):
    ...
    self.als_interaction_count += 1
    if self.als_interaction_count >= self.als_retrain_threshold:
        # Schedule ALS retrain (background task)
        ...
```

#### 3.3 Scheduled Retrain với APScheduler

```python
# server.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    ...
    # Schedule retrain hàng ngày lúc 2 AM
    scheduler.add_job(retrain_models, 'cron', hour=2, minute=0)
    scheduler.start()
```

---

### Phase 4: Concurrency & Persistence (2–3 ngày)

#### 4.1 Thread-safe Interaction Buffer

```python
import threading

class HybridRecommender:
    def __init__(self, ...):
        self._buffer_lock = threading.Lock()
        ...
    
    def add_interaction(self, user_id, book_id, strength, interaction_type):
        with self._buffer_lock:
            self.interaction_buffer.append({...})
            if len(self.interaction_buffer) >= self.buffer_size:
                return self.incremental_update(force=True)
        return False
```

#### 4.2 Persistent Buffer với Redis (Tùy chọn nâng cao)

```python
# Thay vì in-memory list, dùng Redis list để persist buffer
# Đảm bảo buffer không bị mất khi server restart

import redis.asyncio as redis

class PersistentInteractionBuffer:
    def __init__(self, redis_client, key: str = "interaction_buffer"):
        self.redis = redis_client
        self.key = key
    
    async def push(self, interaction: dict):
        await self.redis.rpush(self.key, json.dumps(interaction))
    
    async def pop_all(self) -> List[dict]:
        items = await self.redis.lrange(self.key, 0, -1)
        await self.redis.delete(self.key)
        return [json.loads(i) for i in items]
```

#### 4.3 `asyncio.Lock` cho Retrain Status

```python
# routes.py
import asyncio

_retrain_lock = asyncio.Lock()

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    if _retrain_lock.locked():
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    background_tasks.add_task(retrain_models)
    ...
```

---

### Phase 5: API Improvements & Backend Integration (2 ngày)

#### 5.1 Thêm `progress` field vào FeedbackRequest

```python
class FeedbackRequest(BaseModel):
    user_id: int
    book_id: int
    event: str  # 'rating' | 'history' | 'favorite'
    rating_value: Optional[int] = None
    progress: Optional[float] = None  # 0-100, dùng cho 'history' event
    timestamp: Optional[datetime] = None  # Cho phép backend gửi timestamp thực
```

#### 5.2 Thêm Batch Feedback Endpoint

```python
@router.post("/feedback/batch")
async def record_feedback_batch(requests: List[FeedbackRequest], background_tasks: BackgroundTasks):
    """Gửi nhiều feedback một lúc (efficient hơn cho batch sync từ backend)"""
```

#### 5.3 Thêm Health Check chi tiết

```python
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": recommender is not None,
        "is_retraining": is_retraining,
        "last_trained_at": recommender.trained_at if recommender else None,
        "als_users": len(recommender.als_model.user_id_map) if recommender?.als_model else 0,
    }
```

#### 5.4 Xác Thực Request từ Backend (API Key)

```python
# Hiện tại CORS allow_origins=["*"] — không có auth
# Thêm API key validation cho internal endpoints

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-Internal-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != settings.internal_api_key:
        raise HTTPException(status_code=403)

@router.post("/retrain", dependencies=[Depends(verify_api_key)])
async def trigger_retrain(...):
    ...
```

---

## Tóm Tắt Ưu Tiên

| Mức độ | Vấn đề | Phase |
|---|---|---|
| 🔴 Critical | Race condition khi retrain | Phase 1.3 |
| 🔴 Critical | `_user_in_model` bug → evaluation luôn sai | Phase 1.2 |
| 🔴 Critical | `db_uri` hard-code `localhost` (Docker broken) | Phase 1.1 |
| 🟠 High | Interaction strength logic sai (progress=0 > progress=20%) | Phase 2.1 |
| 🟠 High | Không có model versioning / rollback | Phase 3.1 |
| 🟠 High | `interaction_buffer` không thread-safe | Phase 4.1 |
| 🟡 Medium | ALS stale sau nhiều online update, không có auto-retrain | Phase 3.2 |
| 🟡 Medium | Không có Scheduled Retrain | Phase 3.3 |
| 🟡 Medium | `FeedbackRequest` thiếu `progress` field | Phase 5.1 |
| 🟢 Low | Không có API Key authentication | Phase 5.4 |
| 🟢 Low | Buffer không persistent (mất khi restart) | Phase 4.2 |

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Bạn muốn tôi triển khai tất cả các phase hay chỉ tập trung vào các bug nghiêm trọng (Phase 1 + 2) trước?

> [!IMPORTANT]
> **Q2**: Hệ thống có đang chạy production không? Nếu có, Phase 1 (bug fix) nên được apply trước tiên mà không cần downtime.

> [!NOTE]
> **Q3**: Scheduled retrain (Phase 3.3) — bạn muốn trigger theo cron (hàng ngày) hay theo số lượng interaction tích lũy? Hoặc cả hai?

> [!NOTE]
> **Q4**: Persistent buffer với Redis (Phase 4.2) — bạn có Redis trong stack hiện tại không? Nếu không, có thể dùng SQLite hoặc PostgreSQL thay thế.

## Verification Plan

### Automated Tests
- Chạy `python train.py --evaluate` và verify HR@K > 0 (hiện tại evaluation có thể trả về 0 do bug `_user_in_model`)
- Test `/feedback` endpoint với `event='history'` và `progress=20` → verify strength = 1.0
- Test `/retrain` với concurrent requests → verify 409 response thứ 2

### Manual Verification
- Verify Docker container có thể connect database (sau fix `db_uri`)
- Verify recommendations trước và sau retrain không bị corrupt trong quá trình retrain
