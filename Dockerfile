# Multi-stage Dockerfile for RAG Doctor MCP
# CPU-only base image (GPU variant available separately)

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

# Build wheel
RUN uv build --wheel --no-cache-dir

# Runtime stage
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install package
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Create directories for model cache and collections
RUN mkdir -p /home/appuser/.cache/ragdoc /home/appuser/.ragdoc/collections \
    && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Environment variables
ENV RAGDOC_BASE_DIR=/home/appuser/.ragdoc
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TORCH_HOME=/home/appuser/.cache/torch
ENV FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed

# Expose MCP stdio (no ports for stdio transport)
# For HTTP transport, uncomment:
# EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ragdoc; print('OK')" || exit 1

# Entry point
ENTRYPOINT ["ragdoc"]
CMD []