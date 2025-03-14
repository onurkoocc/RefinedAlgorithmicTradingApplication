# Use TensorFlow 2.16.1 with GPU support as base image
FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    TF_XLA_FLAGS="--tf_xla_auto_jit=2" \
    TF_GPU_THREAD_MODE=gpu_private \
    TF_GPU_THREAD_COUNT=2 \
    TF_ENABLE_GPU_GARBAGE_COLLECTION=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

RUN mkdir -p /app/data /app/results/models /app/results/backtest /app/results/logs

COPY *.py /app/
COPY startup.sh /app/
# Fix Windows line endings by removing \r
RUN sed -i 's/\r$//' /app/startup.sh
RUN chmod +x /app/startup.sh

ENTRYPOINT ["/app/startup.sh"]