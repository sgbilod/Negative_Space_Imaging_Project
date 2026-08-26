# Phase 10: Implementation Starter Kit
## Quick-Start Guides for Priority Technologies

**Last Updated:** February 19, 2026
**Target:** Get first improvements running in Week 1-2

---

## PART 1: TENSORRT OPTIMIZATION (Week 1-3)

### Why TensorRT First?
- ✅ 10x inference speedup (existing models)
- ✅ No model retraining needed
- ✅ Drop-in replacement for PyTorch
- ✅ Immediate ROI
- ✅ Low risk (fallback to PyTorch always available)

### Prerequisites
```bash
# GPU: NVIDIA CUDA Compute Capability 5.3+ (Tesla K40+, GTX 750 Ti+)
# CUDA: 11.8 or higher
# cuDNN: 8.6+
# Python: 3.8+

# Check GPU
nvidia-smi

# Check CUDA
nvcc --version
```

### Installation
```bash
# 1. Install TensorRT
pip install tensorrt==8.6.1

# 2. Install ONNX runtime
pip install onnx onnxruntime-gpu

# 3. Verify
python -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

### Quick Implementation (30 minutes)

**Step 1: Create TensorRT conversion script**
```python
# convert_to_tensorrt.py
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyTorchToTensorRT:
    def __init__(self, model_path, output_dir="trt_models", precision="fp16"):
        """
        Args:
            model_path: Path to PyTorch model
            precision: 'fp32', 'fp16', or 'int8'
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.precision = precision

    def pytorch_to_onnx(self, model, dummy_input, model_name):
        """Convert PyTorch to ONNX"""
        onnx_path = self.output_dir / f"{model_name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )

        logger.info(f"ONNX model saved to {onnx_path}")
        return onnx_path

    def onnx_to_tensorrt(self, onnx_path, model_name, max_batch_size=32):
        """Convert ONNX to TensorRT"""
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)

        # Set precision
        if self.precision == "fp16":
            builder.fp16_mode = True
        elif self.precision == "int8":
            builder.int8_mode = True

        # Parse ONNX model
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Enable dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 640, 640), (16, 3, 640, 640), (32, 3, 640, 640))
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        # Save engine
        trt_path = self.output_dir / f"{model_name}_{self.precision}.trt"
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine saved to {trt_path}")
        return trt_path

    def convert_model(self, model, dummy_input, model_name):
        """Full conversion pipeline"""
        onnx_path = self.pytorch_to_onnx(model, dummy_input, model_name)
        trt_path = self.onnx_to_tensorrt(onnx_path, model_name)
        return trt_path


# Usage
if __name__ == "__main__":
    # Example: Convert YOLO model
    from negative_space_analysis import YOLODetector

    detector = YOLODetector()
    converter = PyTorchToTensorRT(
        model_path="models/yolo.pt",
        precision="fp16"
    )

    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    trt_engine_path = converter.convert_model(
        detector.model,
        dummy_input,
        "yolo_detector"
    )

    print(f"✅ Model converted successfully: {trt_engine_path}")
```

**Step 2: Create TensorRT inference wrapper**
```python
# tensorrt_inference.py
import tensorrt as trt
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        """Load TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)

        # Pre-allocate memory
        self.input_memory = cuda.mem_alloc(np.prod(self.input_shape) * 4)
        self.output_memory = cuda.mem_alloc(np.prod(self.output_shape) * 4)

    def infer(self, input_data):
        """Run inference on TensorRT engine"""
        # Copy input to GPU
        cuda.memcpy_htod(self.input_memory, input_data.astype(np.float32))

        # Execute
        self.context.execute_v2([
            int(self.input_memory),
            int(self.output_memory)
        ])

        # Copy output to CPU
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.output_memory)

        return output

    def __del__(self):
        """Cleanup GPU memory"""
        self.input_memory.free()
        self.output_memory.free()


# Usage
if __name__ == "__main__":
    # Load TensorRT model
    trt_inference = TensorRTInference("trt_models/yolo_detector_fp16.trt")

    # Run inference
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    output = trt_inference.infer(dummy_input)

    print(f"✅ Inference successful: {output.shape}")
```

**Step 3: Integrate into existing pipeline**
```python
# Updated negative_space_analysis/ai_model.py

class NegativeSpaceAnalyzer:
    def __init__(self, use_tensorrt=True):
        self.use_tensorrt = use_tensorrt

        if use_tensorrt:
            try:
                from tensorrt_inference import TensorRTInference
                self.yolo_trt = TensorRTInference("trt_models/yolo_detector_fp16.trt")
                print("✅ Using TensorRT acceleration")
            except:
                print("⚠️ TensorRT not available, falling back to PyTorch")
                self.yolo_trt = None
        else:
            self.yolo_trt = None

    def detect_negative_space(self, image):
        """Enhanced with TensorRT fallback"""
        if self.yolo_trt:
            try:
                # TensorRT inference (10x faster)
                detections = self.yolo_trt.infer(image)
                return detections
            except Exception as e:
                print(f"TensorRT error: {e}, falling back to PyTorch")

        # Fallback to PyTorch
        return self._detect_with_pytorch(image)

    def _detect_with_pytorch(self, image):
        """Original PyTorch implementation"""
        # ... existing code ...
        pass
```

**Step 4: Benchmark**
```bash
# Run benchmarks
python benchmark_tensorrt.py

# Expected output:
# PyTorch FP32:    50.2ms per image
# TensorRT FP32:   12.5ms per image (4.0x faster)
# TensorRT FP16:   8.3ms per image (6.0x faster)
# TensorRT INT8:   2.1ms per image (24x faster!)
```

### Deployment Checklist
- [ ] Benchmark on production GPU
- [ ] Verify accuracy (should be >99% same results)
- [ ] Test fallback mechanism
- [ ] Update docker-compose with CUDA base image
- [ ] Document in README
- [ ] Monitor GPU memory usage
- [ ] Set up Prometheus metrics for latency

---

## PART 2: REDIS CLUSTERING (Week 2-3)

### Why Redis Clustering?
- ✅ 5x cache efficiency
- ✅ High availability (auto-failover)
- ✅ Horizontal scaling
- ✅ Built-in redundancy

### Quick 6-Node Cluster Setup

**Option A: Docker Compose (Easiest)**

Add to `docker-compose.yml`:
```yaml
  redis-node-1:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-node-2:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-node-3:
    image: redis:7-alpine
    ports:
      - "6381:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-node-4:
    image: redis:7-alpine
    ports:
      - "6382:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-node-5:
    image: redis:7-alpine
    ports:
      - "6383:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-node-6:
    image: redis:7-alpine
    ports:
      - "6384:6379"
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    networks:
      - app-network

  redis-cluster-init:
    image: redis:7-alpine
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
      - redis-node-4
      - redis-node-5
      - redis-node-6
    command: >
      redis-cli --cluster create
      redis-node-1:6379 redis-node-2:6379 redis-node-3:6379
      redis-node-4:6379 redis-node-5:6379 redis-node-6:6379
      --cluster-replicas 1
      --cluster-yes
    networks:
      - app-network
```

**Start cluster:**
```bash
docker-compose up -d redis-node-{1..6} redis-cluster-init

# Verify
docker exec redis-node-1 redis-cli -c CLUSTER INFO
```

**Option B: Kubernetes**
```yaml
# k8s/redis-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-cluster-config
  namespace: nsip
data:
  redis.conf: |
    cluster-enabled yes
    cluster-config-file /var/lib/redis/nodes.conf
    cluster-node-timeout 5000
    appendonly yes
    appendfsync everysec

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: nsip
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - /usr/local/etc/redis/redis.conf
        - --bind
        - 0.0.0.0
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        volumeMounts:
        - name: redis-data
          mountPath: /var/lib/redis
        - name: redis-config
          mountPath: /usr/local/etc/redis/
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi

  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

### Application Integration

**Update Python client:**
```python
# cache.py
from rediscluster import RedisCluster
import logging

class CacheManager:
    def __init__(self, cluster_nodes=None):
        if cluster_nodes is None:
            # Default: local cluster
            cluster_nodes = [
                {"host": "localhost", "port": 6379},
                {"host": "localhost", "port": 6380},
                {"host": "localhost", "port": 6381},
                {"host": "localhost", "port": 6382},
                {"host": "localhost", "port": 6383},
                {"host": "localhost", "port": 6384},
            ]

        self.redis = RedisCluster(
            startup_nodes=cluster_nodes,
            decode_responses=True,
            skip_full_coverage_check=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )

        logging.info(f"✅ Connected to Redis Cluster: {self.redis.cluster_info()}")

    def cache_result(self, key, value, ttl=3600):
        """Cache a result with TTL"""
        self.redis.setex(key, ttl, json.dumps(value))

    def get_cached(self, key):
        """Get cached result"""
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

    def invalidate(self, pattern):
        """Invalidate cache by pattern"""
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)


# Usage in NegativeSpaceAnalyzer
class NegativeSpaceAnalyzer:
    def __init__(self):
        self.cache = CacheManager()

    def analyze(self, image_id, image):
        # Check cache first
        cache_key = f"analysis:{image_id}"
        cached = self.cache.get_cached(cache_key)
        if cached:
            return cached

        # Process image
        result = self._process_image(image)

        # Cache result (1 week TTL)
        self.cache.cache_result(cache_key, result, ttl=604800)

        return result
```

### Monitoring
```bash
# Check cluster health
docker exec redis-node-1 redis-cli -c CLUSTER NODES

# Monitor key stats
docker exec redis-node-1 redis-cli INFO stats

# Check replication
docker exec redis-node-1 redis-cli -c INFO replication
```

---

## PART 3: DATABASE OPTIMIZATION (Week 1-3)

### Quick Wins: Strategic Indexes

**Step 1: Analyze current queries**
```bash
# Enable query logging
psql -U postgres -d negative_space -c "
  CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
  ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
"

# Restart PostgreSQL
docker-compose restart postgres

# View slowest queries
psql -U postgres -d negative_space -c "
  SELECT query, calls, mean_exec_time
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 10;
"
```

**Step 2: Add strategic indexes**
```sql
-- Core analysis table indexes
CREATE INDEX CONCURRENTLY idx_images_confidence
  ON images(confidence DESC)
  WHERE confidence > 0.8;

CREATE INDEX CONCURRENTLY idx_images_created_at
  ON images(created_at DESC NULLS LAST);

CREATE INDEX CONCURRENTLY idx_images_device_created
  ON images(device_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_analysis_results_image_id
  ON analysis_results(image_id);

-- Composite index for common query patterns
CREATE INDEX CONCURRENTLY idx_images_device_confidence
  ON images(device_id, confidence DESC)
  WHERE status = 'processed';

-- Partial index for recent data (faster)
CREATE INDEX CONCURRENTLY idx_images_recent
  ON images(id, confidence)
  WHERE created_at > CURRENT_DATE - INTERVAL '30 days';
```

**Step 3: Connection Pooling (PgBouncer)**

Add to `docker-compose.yml`:
```yaml
  pgbouncer:
    image: edoburu/pgbouncer:latest
    environment:
      DATABASE_URL: "postgres://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}"
      PGBOUNCER_POOL_MODE: transaction
      PGBOUNCER_MAX_CLIENT_CONN: 1000
      PGBOUNCER_DEFAULT_POOL_SIZE: 25
      PGBOUNCER_MIN_POOL_SIZE: 5
      PGBOUNCER_RESERVE_POOL_SIZE: 5
      PGBOUNCER_RESERVE_POOL_TIMEOUT: 3
    ports:
      - "6432:6432"
    depends_on:
      - postgres
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "${DB_USER}", "-h", "localhost", "-p", "6432"]
      interval: 10s
      timeout: 5s
      retries: 5
```

Update connection string: `postgresql://user:pass@pgbouncer:6432/database`

**Step 4: Query Tuning**

```sql
-- Analyze query plan
EXPLAIN ANALYZE
SELECT COUNT(*) as total_processed,
       AVG(confidence) as avg_confidence
FROM images
WHERE created_at > NOW() - INTERVAL '7 days'
  AND status = 'processed';

-- Expected: Index Scan (not Sequential Scan)
-- If Sequential Scan → Query needs optimization
```

---

## PART 4: INTEGRATION INTO DOCKER-COMPOSE

Update `docker-compose.yml`:

```yaml
version: "3.8"

services:
  # ... existing services ...

  # TensorRT Optimization: Use cuda base image
  analyzer:
    build:
      context: .
      dockerfile: Dockerfile.python
      args:
        CUDA_VERSION: "11.8"
        TENSORRT_VERSION: "8.6.1"
    environment:
      - TENSORRT_ENABLED=true
      - GPU_MEMORY_FRACTION=0.8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Redis Cluster nodes
  redis-node-1:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --bind 0.0.0.0
    ports:
      - "6379:6379"
    networks:
      - app-network

  # ... 5 more redis nodes ...

  # PgBouncer for connection pooling
  pgbouncer:
    image: edoburu/pgbouncer:latest
    environment:
      DATABASE_URL: "postgres://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}"
      PGBOUNCER_MAX_CLIENT_CONN: 1000
    ports:
      - "6432:6432"
    depends_on:
      - postgres
    networks:
      - app-network

  # Update API to use pgbouncer
  api:
    environment:
      - DATABASE_URL=postgresql://user:pass@pgbouncer:6432/database
      - REDIS_CLUSTER_NODES=redis-node-1:6379,redis-node-2:6379,...
    depends_on:
      - pgbouncer
      - redis-node-1
```

---

## PART 5: PERFORMANCE VALIDATION

### Benchmark Script
```python
# benchmark_improvements.py
import time
import numpy as np
from negative_space_analysis import NegativeSpaceAnalyzer

def benchmark():
    analyzer = NegativeSpaceAnalyzer(use_tensorrt=True)

    # Generate test image
    test_image = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup
    for _ in range(3):
        analyzer.analyze(test_image)

    # Benchmark
    times = []
    for i in range(100):
        start = time.time()
        result = analyzer.analyze(test_image)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # milliseconds

    times = np.array(times)

    print("=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"Average:   {times.mean():.2f}ms")
    print(f"Median:    {np.median(times):.2f}ms")
    print(f"P95:       {np.percentile(times, 95):.2f}ms")
    print(f"P99:       {np.percentile(times, 99):.2f}ms")
    print(f"Min:       {times.min():.2f}ms")
    print(f"Max:       {times.max():.2f}ms")
    print(f"Std Dev:   {times.std():.2f}ms")
    print("=" * 60)
    print(f"✅ Throughput: {1000/times.mean():.1f} images/second")

if __name__ == "__main__":
    benchmark()
```

---

## WEEK 1-3 CHECKLIST

### Week 1: Foundation
- [ ] Install TensorRT, build conversion script
- [ ] Convert 3 main models (YOLO, SegFormer, Pattern Recognition)
- [ ] Benchmark TensorRT models
- [ ] Set up Redis cluster (Docker Compose)
- [ ] Create database indexes
- [ ] Update docker-compose.yml

### Week 2: Integration
- [ ] Integrate TensorRT into inference pipeline
- [ ] Test fallback mechanisms
- [ ] Deploy Redis cluster to staging
- [ ] Update Python client for Redis cluster
- [ ] Deploy PgBouncer
- [ ] Run performance benchmarks

### Week 3: Validation & Deployment
- [ ] Validate accuracy (TensorRT vs PyTorch)
- [ ] Load test (1000+ concurrent requests)
- [ ] Monitor GPU memory usage
- [ ] Document performance improvements
- [ ] Deploy to production
- [ ] Monitor production metrics

### Expected Results
```
Metric           Before    After     Improvement
─────────────────────────────────────────────────
Inference Time   50ms      5ms       10x faster
Throughput       20 img/s  200 img/s 10x higher
Cache Hit Ratio  60%       85%       25% higher
Query Time       500ms     50ms      10x faster
API Response     1000ms    100ms     10x faster
GPU Memory       8GB       2GB       4x reduction
```

---

**Next Steps:** After completing these three, move to Part 2 technologies (Kafka, YOLO, etc.)

---
