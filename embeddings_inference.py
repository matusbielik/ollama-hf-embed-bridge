#!/usr/bin/env python3
"""
Python persistent worker for generating embeddings using any HuggingFace model.
Communicates with Go application via stdin/stdout for high-performance processing.
"""

import json
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List
import traceback
import signal
import warnings

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Configure logging to stderr to avoid interfering with stdout communication
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingGenerator:
    def __init__(self, model_names: str = None):
        # Parse comma-separated model names from environment variable or default
        if model_names is None:
            model_names = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Split comma-separated model names and clean whitespace
        self.model_names = [name.strip() for name in model_names.split(",") if name.strip()]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Worker starting with device: {self.device}")
        
        # Load all models into memory
        self.models = {}
        self.tokenizers = {}
        
        for model_name in self.model_names:
            try:
                logger.info(f"Loading model: {model_name}")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_name] = AutoModel.from_pretrained(model_name)
                self.models[model_name].to(self.device)
                self.models[model_name].eval()
                logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        logger.info(f"All {len(self.model_names)} models loaded successfully - worker ready")
    
    def generate_embeddings(self, texts: List[str], model_name: str = None) -> List[List[float]]:
        """Generate embeddings for a list of texts using specified model."""
        if not texts:
            return []
        
        # Use first model as default if no model specified
        if model_name is None:
            model_name = self.model_names[0]
        
        # Validate model name
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        embeddings = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model outputs
                    outputs = model(**inputs)
                    
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Apply attention mask and mean pool
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    masked_embeddings = token_embeddings * input_mask_expanded
                    sum_embeddings = torch.sum(masked_embeddings, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask
                    
                    # Normalize
                    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                    
                    # Convert to list and ensure clean floats
                    embedding = normalized.cpu().numpy().flatten()
                    # Replace any NaN or inf values with 0.0
                    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
                    embeddings.append(embedding.tolist())
            
            logger.info(f"Generated embeddings for {len(texts)} texts using model {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with model {model_name}: {e}")
            raise

def main():
    """Main persistent worker loop."""
    try:
        # Initialize generator once at startup
        generator = HuggingFaceEmbeddingGenerator()
        
        # Send ready signal to Go process with all available models
        ready_response = {
            "status": "ready",
            "models": generator.model_names,
            "device": str(generator.device)
        }
        print(json.dumps(ready_response), flush=True)
        
        # Process requests in a loop
        while True:
            try:
                # Read line from stdin (blocks until Go sends request)
                line = sys.stdin.readline()
                if not line:  # EOF - Go process closed
                    logger.info("Received EOF, shutting down worker")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    error_response = {"error": f"Invalid JSON: {str(e)}"}
                    print(json.dumps(error_response), flush=True)
                    continue
                
                # Handle different request types
                if request.get("type") == "ping":
                    # Health check
                    response = {"status": "pong"}
                    print(json.dumps(response), flush=True)
                    continue
                
                if request.get("type") == "shutdown":
                    # Graceful shutdown
                    logger.info("Received shutdown request")
                    break
                
                # Default: embedding request
                texts = request.get('texts', [])
                model_name = request.get('model')  # Extract requested model
                
                if not texts:
                    error_response = {"error": "No texts provided"}
                    print(json.dumps(error_response), flush=True)
                    continue
                
                # Generate embeddings with specified model
                try:
                    embeddings = generator.generate_embeddings(texts, model_name)
                    used_model = model_name if model_name else generator.model_names[0]
                except ValueError as e:
                    error_response = {"error": str(e)}
                    print(json.dumps(error_response), flush=True)
                    continue
                
                # Send response
                response = {
                    "embeddings": embeddings,
                    "model": used_model,
                    "device": str(generator.device)
                }
                
                # Validate JSON before sending
                try:
                    response_json = json.dumps(response)
                    print(response_json, flush=True)
                except (TypeError, ValueError) as e:
                    error_response = {"error": f"JSON serialization failed: {str(e)}"}
                    print(json.dumps(error_response), flush=True)
                
            except Exception as e:
                # Handle per-request errors without crashing worker
                logger.error(f"Request processing error: {e}")
                logger.error(traceback.format_exc())
                error_response = {"error": str(e)}
                print(json.dumps(error_response), flush=True)
        
        logger.info("Worker shutting down normally")
        
    except Exception as e:
        logger.error(f"Worker fatal error: {e}")
        logger.error(traceback.format_exc())
        error_response = {"error": f"Worker fatal error: {str(e)}"}
        print(json.dumps(error_response), flush=True)
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down worker")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    main()