# Base image
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# OS dependencies (scikit-learn needs OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port (Azure may override with PORT env var)
EXPOSE 8000

# Optional: allow tuning via env vars GUNICORN_WORKERS and GUNICORN_THREADS
# Default to 4 workers and 8 threads per worker if not provided
CMD ["sh", "-c", "gunicorn -k gthread -w ${GUNICORN_WORKERS:-4} --threads ${GUNICORN_THREADS:-8} -b 0.0.0.0:${PORT} app:app"]
