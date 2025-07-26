# Multi-stage build for optimized image size
FROM golang:1.24.5-alpine AS go-builder

# Install git for go modules
RUN apk add --no-cache git

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o ollama-hf-bridge .

# Final image with Python and Go binary
FROM python:3.12-slim

# Build argument for model selection
ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV MODEL_NAME=${MODEL_NAME}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python inference script
COPY embeddings_inference.py .
RUN chmod +x embeddings_inference.py

# Copy Go binary from builder stage
COPY --from=go-builder /app/ollama-hf-bridge .

# Create non-root user for security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Pre-download model to avoid first-run delay (optional - uncomment if desired)
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('${MODEL_NAME}'); AutoModel.from_pretrained('${MODEL_NAME}')"

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/ || exit 1

# Default command
CMD ["./ollama-hf-bridge", "-host", "0.0.0.0", "-port", "11434"]
