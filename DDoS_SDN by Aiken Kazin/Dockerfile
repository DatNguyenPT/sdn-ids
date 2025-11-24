# Flower Server Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-server.txt .

# Install Python dependencies (add requests for dashboard communication)
RUN pip install --no-cache-dir -r requirements-server.txt && \
    pip install --no-cache-dir requests>=2.25.0

# Copy server scripts
COPY flower_server.py .
COPY flower_server_metrics.py .

# Expose Flower server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('localhost', 8080)); s.close()" || exit 1

# Run Flower server
CMD ["python", "flower_server.py", "--address", "0.0.0.0", "--port", "8080"]

