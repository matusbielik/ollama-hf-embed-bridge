package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/matusbielik/ollama-hf-embed-bridge/model"
)

type Server struct {
	embeddingModel *model.EmbeddingModel
	host          string
	port          string
}

type EmbedRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
}

type EmbedRequestAlt struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

type TagsResponse struct {
	Models []ModelInfo `json:"models"`
}

type ModelInfo struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Details    Details   `json:"details"`
}

type Details struct {
	Family    string `json:"family"`
	Families  []string `json:"families"`
	Format    string `json:"format"`
	ParamSize string `json:"parameter_size"`
}

func main() {
	host := flag.String("host", "localhost", "Host to bind to")
	port := flag.String("port", "11434", "Port to bind to")
	flag.Parse()

	log.Printf("Starting Ollama HuggingFace Bridge server...")
	log.Printf("Downloading and loading model...")

	files, err := model.DownloadModel()
	if err != nil {
		log.Fatalf("Failed to download model: %v", err)
	}

	embeddingModel, err := model.LoadModel(files)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	server := &Server{
		embeddingModel: embeddingModel,
		host:          *host,
		port:          *port,
	}

	server.start()
}

func (s *Server) start() {
	mux := http.NewServeMux()
	
	mux.HandleFunc("/api/embed", s.handleEmbed)
	mux.HandleFunc("/api/embeddings", s.handleEmbeddingsAlt)
	mux.HandleFunc("/api/tags", s.handleTags)
	mux.HandleFunc("/", s.handleRoot)

	httpServer := &http.Server{
		Addr:    fmt.Sprintf("%s:%s", s.host, s.port),
		Handler: mux,
	}

	go func() {
		log.Printf("Server listening on http://%s:%s", s.host, s.port)
		log.Printf("Endpoints:")
		log.Printf("  POST /api/embed")
		log.Printf("  POST /api/embeddings")
		log.Printf("  GET  /api/tags")
		
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	<-c
	log.Println("Shutting down server...")

	// Shutdown worker pool first
	s.embeddingModel.Shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	} else {
		log.Println("Server gracefully stopped")
	}
}

func (s *Server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	texts, err := extractTexts(req.Input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	embeddings, err := s.embeddingModel.GenerateEmbeddings(texts)
	if err != nil {
		log.Printf("Error generating embeddings: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	response := EmbedResponse{
		Model:      "small-e-czech",
		Embeddings: embeddings,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleEmbeddingsAlt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbedRequestAlt
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	texts := []string{req.Prompt}
	embeddings, err := s.embeddingModel.GenerateEmbeddings(texts)
	if err != nil {
		log.Printf("Error generating embeddings: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	response := EmbedResponse{
		Model:      "small-e-czech",
		Embeddings: embeddings,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	response := TagsResponse{
		Models: []ModelInfo{
			{
				Name:       "small-e-czech:latest",
				Model:      "small-e-czech",
				ModifiedAt: time.Now(),
				Size:       67108864, // ~64MB
				Details: Details{
					Family:    "electra",
					Families:  []string{"electra"},
					Format:    "pytorch",
					ParamSize: "15M",
				},
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "Ollama HuggingFace Bridge\n")
	fmt.Fprintf(w, "Run any HuggingFace embedding model with Ollama API\n")
	fmt.Fprintf(w, "Compatible with Ollama clients\n\n")
	fmt.Fprintf(w, "Endpoints:\n")
	fmt.Fprintf(w, "  POST /api/embed\n")
	fmt.Fprintf(w, "  POST /api/embeddings\n")
	fmt.Fprintf(w, "  GET  /api/tags\n")
}

func extractTexts(input interface{}) ([]string, error) {
	switch v := input.(type) {
	case string:
		return []string{v}, nil
	case []interface{}:
		texts := make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				texts[i] = str
			} else {
				return nil, fmt.Errorf("input array must contain only strings")
			}
		}
		return texts, nil
	default:
		return nil, fmt.Errorf("input must be a string or array of strings")
	}
}