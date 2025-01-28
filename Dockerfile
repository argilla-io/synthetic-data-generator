# Use Python slim image as base
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create and set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Gradio port
EXPOSE 7860

# Set default environment variables for configuration
ENV MODEL="meta-llama/Llama-3.1-8B-Instruct" \
    MAGPIE_PRE_QUERY_TEMPLATE="llama3" \
    MAX_NUM_TOKENS=2048 \
    MAX_NUM_ROWS=1000 \
    DEFAULT_BATCH_SIZE=5

# Start command
CMD ["python", "-m", "synthetic_dataset_generator"] 