# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD ["python", "server.py"]


FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set dashboard port
ENV FLOWER_DASHBOARD_PORT=8081

# Chạy SuperNode
CMD ["flower-supernode", "--insecure"]