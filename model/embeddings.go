package model

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

type EmbeddingModel struct {
	workerPool *WorkerPool
	modelName  string
}

type WorkerPool struct {
	workers      []*Worker
	requestChan  chan *WorkerRequest
	shutdownChan chan bool
	mu           sync.RWMutex
	pythonScript string
	numWorkers   int
}

type Worker struct {
	id       int
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdout   *bufio.Scanner
	stderr   io.ReadCloser
	ready    bool
	mu       sync.Mutex
}

type WorkerRequest struct {
	Texts        []string `json:"texts"`
	Type         string   `json:"type,omitempty"`
	ResponseChan chan *WorkerResponse `json:"-"` // Don't marshal this field
}

type WorkerResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Device     string      `json:"device"`
	Status     string      `json:"status,omitempty"`
	Error      string      `json:"error,omitempty"`
}

func LoadModel(files *ModelFiles) (*EmbeddingModel, error) {
	// Find Python script path
	scriptPath := "embeddings_inference.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		// Try absolute path
		wd, _ := os.Getwd()
		scriptPath = filepath.Join(wd, "embeddings_inference.py")
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("Python script not found: %s", scriptPath)
		}
	}

	// Create worker pool with 2 workers (can be configurable)
	numWorkers := 2
	workerPool, err := NewWorkerPool(scriptPath, numWorkers)
	if err != nil {
		return nil, fmt.Errorf("failed to create worker pool: %w", err)
	}

	model := &EmbeddingModel{
		workerPool: workerPool,
		modelName:  "Seznam/small-e-czech",
	}

	log.Printf("Initialized persistent worker pool with %d workers for model: %s", numWorkers, model.modelName)
	return model, nil
}

func NewWorkerPool(scriptPath string, numWorkers int) (*WorkerPool, error) {
	pool := &WorkerPool{
		workers:      make([]*Worker, 0, numWorkers),
		requestChan:  make(chan *WorkerRequest, numWorkers*2), // Buffered channel
		shutdownChan: make(chan bool),
		pythonScript: scriptPath,
		numWorkers:   numWorkers,
	}

	// Start workers
	for i := 0; i < numWorkers; i++ {
		worker, err := pool.startWorker(i)
		if err != nil {
			// Cleanup any started workers
			pool.Shutdown()
			return nil, fmt.Errorf("failed to start worker %d: %w", i, err)
		}
		pool.workers = append(pool.workers, worker)
	}

	// Start request dispatcher
	go pool.dispatcher()

	log.Printf("Worker pool initialized with %d workers", numWorkers)
	return pool, nil
}

func (pool *WorkerPool) startWorker(id int) (*Worker, error) {
	// Try to use virtual environment first, fallback to system python3
	pythonCmd := "venv/bin/python3"
	if _, err := os.Stat(pythonCmd); os.IsNotExist(err) {
		pythonCmd = "python3"
	}
	
	cmd := exec.Command(pythonCmd, pool.pythonScript)
	
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	err = cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("failed to start python process: %w", err)
	}

	// Create scanner with larger buffer for big embedding responses
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 10*1024*1024) // 10MB max token size
	
	worker := &Worker{
		id:     id,
		cmd:    cmd,
		stdin:  stdin,
		stdout: scanner,
		stderr: stderr,
		ready:  false,
	}

	// Wait for ready signal
	if !worker.stdout.Scan() {
		return nil, fmt.Errorf("failed to read ready signal from worker %d", id)
	}

	var readyResponse WorkerResponse
	err = json.Unmarshal(worker.stdout.Bytes(), &readyResponse)
	if err != nil || readyResponse.Status != "ready" {
		return nil, fmt.Errorf("worker %d did not send ready signal: %v", id, err)
	}

	worker.ready = true
	log.Printf("Worker %d ready on device: %s", id, readyResponse.Device)

	// Start stderr monitoring goroutine
	go worker.monitorStderr()

	return worker, nil
}

func (pool *WorkerPool) dispatcher() {
	for {
		select {
		case req := <-pool.requestChan:
			// Find available worker and process request
			go pool.processRequest(req)
		case <-pool.shutdownChan:
			return
		}
	}
}

func (pool *WorkerPool) processRequest(req *WorkerRequest) {
	// Get an available worker (simple round-robin for now)
	pool.mu.RLock()
	if len(pool.workers) == 0 {
		pool.mu.RUnlock()
		req.ResponseChan <- &WorkerResponse{Error: "no workers available"}
		return
	}
	
	// Simple worker selection (can be improved with load balancing)
	worker := pool.workers[0]
	for _, w := range pool.workers {
		if w.ready {
			worker = w
			break
		}
	}
	pool.mu.RUnlock()

	// Send request to worker
	response := worker.processRequest(req)
	req.ResponseChan <- response
}

func (worker *Worker) processRequest(req *WorkerRequest) *WorkerResponse {
	worker.mu.Lock()
	defer worker.mu.Unlock()

	// Send request
	requestJSON, err := json.Marshal(req)
	if err != nil {
		return &WorkerResponse{Error: fmt.Sprintf("failed to marshal request: %v", err)}
	}

	_, err = worker.stdin.Write(append(requestJSON, '\n'))
	if err != nil {
		return &WorkerResponse{Error: fmt.Sprintf("failed to send request to worker: %v", err)}
	}

	// Read response
	if !worker.stdout.Scan() {
		return &WorkerResponse{Error: "failed to read response from worker"}
	}

	var response WorkerResponse
	err = json.Unmarshal(worker.stdout.Bytes(), &response)
	if err != nil {
		return &WorkerResponse{Error: fmt.Sprintf("failed to parse worker response: %v", err)}
	}

	return &response
}

func (worker *Worker) monitorStderr() {
	scanner := bufio.NewScanner(worker.stderr)
	for scanner.Scan() {
		log.Printf("Worker %d stderr: %s", worker.id, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("Worker %d stderr monitoring error: %v", worker.id, err)
	}
}

func (pool *WorkerPool) Shutdown() {
	pool.mu.Lock()
	defer pool.mu.Unlock()

	close(pool.shutdownChan)

	// Shutdown all workers
	for _, worker := range pool.workers {
		// Send shutdown signal
		shutdownReq := WorkerRequest{Type: "shutdown"}
		requestJSON, _ := json.Marshal(shutdownReq)
		worker.stdin.Write(append(requestJSON, '\n'))
		
		// Close pipes and wait for process
		worker.stdin.Close()
		worker.cmd.Wait()
	}

	log.Printf("Worker pool shut down")
}

func (em *EmbeddingModel) GenerateEmbeddings(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Create request with response channel
	responseChan := make(chan *WorkerResponse, 1)
	request := &WorkerRequest{
		Texts:        texts,
		ResponseChan: responseChan,
	}

	// Send request to worker pool
	select {
	case em.workerPool.requestChan <- request:
		// Request queued successfully
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("request timeout: worker pool is busy")
	}

	// Wait for response
	select {
	case response := <-responseChan:
		if response.Error != "" {
			return nil, fmt.Errorf("worker error: %s", response.Error)
		}

		if len(response.Embeddings) != len(texts) {
			return nil, fmt.Errorf("unexpected response length: got %d, expected %d", len(response.Embeddings), len(texts))
		}

		log.Printf("Generated embeddings for %d texts using persistent worker on %s", len(texts), response.Device)
		return response.Embeddings, nil

	case <-time.After(30 * time.Second):
		return nil, fmt.Errorf("worker response timeout")
	}
}

func (em *EmbeddingModel) Shutdown() {
	if em.workerPool != nil {
		em.workerPool.Shutdown()
	}
}

