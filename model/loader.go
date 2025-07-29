package model

import (
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// Model registry and available models
var AvailableModels []string

type ModelFiles struct {
	ConfigPath     string
	TokenizerPath  string
	ModelPath      string
	VocabPath      string
}

func GetModelsDir() string {
	homeDir, _ := os.UserHomeDir()
	return filepath.Join(homeDir, ".cache", "czech-embeddings")
}

// InitializeModels parses MODEL_NAME env var and initializes the available models registry
func InitializeModels() error {
	modelNames := os.Getenv("MODEL_NAME")
	if modelNames == "" {
		modelNames = "sentence-transformers/all-MiniLM-L6-v2"
	}
	
	// Parse comma-separated model names
	for _, modelName := range strings.Split(modelNames, ",") {
		modelName = strings.TrimSpace(modelName)
		if modelName != "" {
			AvailableModels = append(AvailableModels, modelName)
		}
	}
	
	log.Printf("Initialized model registry with %d models: %v", len(AvailableModels), AvailableModels)
	return nil
}

// GetAvailableModels returns the list of available models
func GetAvailableModels() []string {
	return AvailableModels
}

// DownloadModel downloads a specific model (legacy function, kept for compatibility)
func DownloadModel() (*ModelFiles, error) {
	// This function is now primarily used for compatibility
	// The actual model loading is handled by Python workers
	files := &ModelFiles{}
	return files, nil
}

func downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func verifyFileIntegrity(filepath string, expectedHash string) error {
	file, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return err
	}

	actualHash := fmt.Sprintf("%x", hasher.Sum(nil))
	if actualHash != expectedHash {
		return fmt.Errorf("file integrity check failed: expected %s, got %s", expectedHash, actualHash)
	}

	return nil
}