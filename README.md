# Ollama HuggingFace Bridge

Ollama-compatible API server for **any HuggingFace embedding model** with **local inference** and **persistent workers**.

> **⚠️ Embedding Models Only**: This project implements Ollama's embedding endpoints (`/api/embed`, `/api/embeddings`) for vector generation. It does **not** support text generation, chat, or conversational models. For full Ollama functionality with generative models, use [Ollama](https://ollama.com) directly.

## Features

- 🤗 **Any HuggingFace Model**: Use any embedding model from HuggingFace Hub
- ⚡ **Persistent Workers**: 100x+ performance improvement over process spawning
- 🔌 **Ollama Compatible**: Drop-in replacement for Ollama **embedding endpoints only**
- 🐳 **Production Ready**: Docker support with configurable models
- 🚀 **GPU Support**: Automatic CUDA detection, CPU fallback
- 🌐 **No API Keys**: Completely local inference

## Use Cases

Perfect for **embedding-focused applications**:
- **Vector Search**: MongoDB Vector Search, Elasticsearch, Pinecone, Weaviate
- **RAG Systems**: Document/query embedding for retrieval-augmented generation
- **Semantic Search**: Content similarity and search applications
- **ML Pipelines**: Embedding generation for downstream ML tasks

**Not suitable for**: Text generation, chatbots, or conversational AI (use [Ollama](https://ollama.com) for that).

## Why This Project?

**Fills the gap between Ollama and HuggingFace for embeddings:**

| Feature | Ollama | Ollama HF Bridge | HuggingFace Direct |
|---------|---------|------------------|-------------------|
| **Embedding Models** | Limited selection | Any HF model | Any HF model |
| **Ollama API Compatible** | ✅ | ✅ | ❌ |
| **Local Inference** | ✅ | ✅ | ✅ |
| **Model Conversion Required** | ✅ (GGUF) | ❌ | ❌ |
| **Production Performance** | ✅ | ✅ (Persistent workers) | ❌ (Process per request) |
| **Text Generation** | ✅ | ❌ | ✅ |
| **Specialized Embeddings** | Limited | ✅ (Any domain/language) | ✅ |

## Docker Usage (Recommended)

### Prerequisites

For **GPU acceleration** (recommended for better performance):
- NVIDIA GPU with recent drivers (550+)
- NVIDIA Container Runtime installed:
  ```bash
  # Add NVIDIA repository (example for Debian / Ubuntu)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

### Build with Default Model
```bash
docker build -t ollama-hf-embed-bridge .

# CPU-only inference
docker run -p 11434:11434 ollama-hf-embed-bridge

# GPU-accelerated inference (recommended)
docker run -p 11434:11434 --gpus all ollama-hf-embed-bridge
```

### Build with Custom Model
```bash
# Popular English models
docker build --build-arg MODEL_NAME="sentence-transformers/all-mpnet-base-v2" -t ollama-hf-embed-bridge .
docker build --build-arg MODEL_NAME="intfloat/e5-large-v2" -t ollama-hf-embed-bridge .

# Multilingual models  
docker build --build-arg MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" -t ollama-hf-embed-bridge .

# Czech models (keeping the original use case!)
docker build --build-arg MODEL_NAME="Seznam/small-e-czech" -t ollama-hf-embed-bridge .
docker build --build-arg MODEL_NAME="Seznam/simcse-small-e-czech" -t ollama-hf-embed-bridge .

# Run with custom model (CPU-only)
docker run -p 11434:11434 ollama-hf-embed-bridge

# Run with custom model (GPU-accelerated)
docker run -p 11434:11434 --gpus all ollama-hf-embed-bridge
```

### Runtime Model Override
```bash
# Override any model at runtime (downloads on first use)
docker run -p 11434:11434 --gpus all -e MODEL_NAME="BAAI/bge-large-en-v1.5" ollama-hf-embed-bridge
```

### GPU Support Details

The container automatically detects and uses GPU when available:
- **Requirements**: NVIDIA GPU with driver 550+ and NVIDIA Container Runtime
- **Compatibility**: RTX 20xx, 30xx, 40xx series and newer
- **Fallback**: Automatically uses CPU if GPU is unavailable or `--gpus all` flag is omitted
- **Performance**: 5-10x faster inference with GPU acceleration

**Test GPU availability**:
```bash
docker run --rm --gpus all ollama-hf-embed-bridge python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Local Installation (Advanced)

> **💡 Docker is recommended** for most users. Use local installation only if you need to modify the code or have specific requirements.

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build and Run
```bash
# Build
go build -o ollama-hf-bridge .

# Run (default: localhost:11434)
./ollama-hf-bridge

# Run on custom host/port  
./ollama-hf-bridge -host 0.0.0.0 -port 8080
```

## API Endpoints

### POST /api/embed
```bash
# English example (default model)
curl http://localhost:11434/api/embed -d '{
  "model": "all-MiniLM-L6-v2",
  "input": ["Hello world", "How are you?"]
}'

# Czech example
curl http://localhost:11434/api/embed -d '{
  "model": "small-e-czech",
  "input": ["Dobrý den", "Jak se máte?"]
}'

# Multilingual example
curl http://localhost:11434/api/embed -d '{
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "input": ["Hello", "Bonjour", "Hola", "Ahoj"]
}'
```

### POST /api/embeddings  
```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "all-MiniLM-L6-v2", 
  "prompt": "This is a test sentence"
}'
```

### GET /api/tags
```bash
curl http://localhost:11434/api/tags
```

## Drop-in Ollama Replacement

This server is compatible with existing Ollama clients:

```python
import ollama

# Default port
client = ollama.Client(host='http://localhost:11434')

# Or custom port
client = ollama.Client(host='http://localhost:8080')

response = client.embeddings(model='all-MiniLM-L6-v2', prompt='Hello world!')

# Works with any model you've configured
response = client.embeddings(model='small-e-czech', prompt='Ahoj světe!')
```

## Supported Models

Works with **any HuggingFace embedding model**:

### Popular English Models
- `sentence-transformers/all-MiniLM-L6-v2` (default)
- `sentence-transformers/all-mpnet-base-v2` 
- `intfloat/e5-large-v2`
- `BAAI/bge-large-en-v1.5`

### Multilingual Models
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/LaBSE`

### Czech Models (Original Use Case)
- `Seznam/small-e-czech` - Czech ELECTRA fine-tuned with SimCSE
- `Seznam/simcse-small-e-czech` 
- `Seznam/dist-mpnet-czeng-cs-en`

## Architecture

- **Go HTTP Server**: High-performance API server with Ollama compatibility
- **Python Workers**: Persistent processes for model inference using PyTorch
- **Process Pool**: Multiple workers for concurrent request handling
- **Docker**: Production-ready containerization with configurable models

## Contributing

We welcome contributions! This project was collaboratively developed and benefits from community input.

## License

MIT License - see [LICENSE](LICENSE) for details.