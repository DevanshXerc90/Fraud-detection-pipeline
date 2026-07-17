FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]