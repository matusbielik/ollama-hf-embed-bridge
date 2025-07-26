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

const (
	ModelRepo = "Seznam/small-e-czech"
	ModelURL  = "https://huggingface.co/" + ModelRepo + "/resolve/main/"
)

type ModelFiles struct {
	ConfigPath     string
	TokenizerPath  string
	ModelPath      string
	VocabPath      string
}

func (m *ModelFiles) ModelsDir() string {
	homeDir, _ := os.UserHomeDir()
	return filepath.Join(homeDir, ".cache", "czech-embeddings", ModelRepo)
}

func DownloadModel() (*ModelFiles, error) {
	files := &ModelFiles{}
	modelsDir := files.ModelsDir()
	
	err := os.MkdirAll(modelsDir, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create models directory: %w", err)
	}

	requiredFiles := map[string]*string{
		"config.json":         &files.ConfigPath,
		"tokenizer.json":      &files.TokenizerPath,
		"pytorch_model.bin":   &files.ModelPath,
		"vocab.txt":           &files.VocabPath,
	}

	for filename, pathPtr := range requiredFiles {
		*pathPtr = filepath.Join(modelsDir, filename)
		
		if fileExists(*pathPtr) {
			log.Printf("Model file %s already exists, skipping download", filename)
			continue
		}
		
		log.Printf("Downloading %s...", filename)
		err := downloadFile(ModelURL+filename, *pathPtr)
		if err != nil {
			log.Printf("Failed to download %s: %v", filename, err)
			// Try alternative files for tokenizer
			if strings.Contains(filename, "tokenizer") {
				altFile := "tokenizer_config.json"
				altPath := filepath.Join(modelsDir, altFile)
				*pathPtr = altPath
				if !fileExists(altPath) {
					err = downloadFile(ModelURL+altFile, altPath)
					if err != nil {
						return nil, fmt.Errorf("failed to download %s or %s: %w", filename, altFile, err)
					}
				}
			} else {
				return nil, fmt.Errorf("failed to download required file %s: %w", filename, err)
			}
		}
	}

	log.Printf("Model downloaded successfully to %s", modelsDir)
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