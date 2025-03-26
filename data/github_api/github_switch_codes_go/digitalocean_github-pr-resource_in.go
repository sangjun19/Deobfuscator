// Repository: digitalocean/github-pr-resource
// File: in.go

package resource

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	m "github.com/digitalocean/concourse-resource-library/metadata"
)

// Get (business logic)
func Get(request GetRequest, github Github, git Git, outputDir string) (*GetResponse, error) {
	if request.Params.SkipDownload {
		return &GetResponse{Version: request.Version}, nil
	}

	pull, err := github.GetPullRequest(request.Version.PR, request.Version.Commit)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve pull request: %s", err)
	}

	// Initialize and pull the base for the PR
	err = git.Clone(pull.RepositoryURL, pull.BaseRefName, request.Params.GitDepth)
	if err != nil {
		return nil, fmt.Errorf("failed to clone repository: %s", err)
	}

	// Fetch the PR and merge the specified commit into the base
	if err := git.Fetch(pull.Number, request.Params.GitDepth); err != nil {
		return nil, err
	}

	switch request.Params.IntegrationTool {
	case "rebase":
		pull.BaseRefOID, err = git.RevParse(pull.BaseRefName)
		if err != nil {
			return nil, err
		}

		if err := git.Rebase(pull.BaseRefName, request.Version.Commit); err != nil {
			return nil, err
		}
	case "merge":
		pull.BaseRefOID, err = git.RevParse(pull.BaseRefName)
		if err != nil {
			return nil, err
		}

		if err := git.Merge(request.Version.Commit); err != nil {
			return nil, err
		}
	case "checkout", "":
		if err := git.Checkout(pull.HeadRefName, request.Version.Commit); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("invalid integration tool specified: %s", request.Params.IntegrationTool)
	}

	if request.Source.GitCryptKey != "" {
		if err := git.GitCryptUnlock(request.Source.GitCryptKey); err != nil {
			return nil, err
		}
	}

	path := filepath.Join(outputDir, ".git", "resource")
	if err := os.MkdirAll(path, os.ModePerm); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %s", err)
	}

	metadata := metadataFactory(pull)
	metadata.AddJSON("version", &request.Version)

	b, err := metadata.JSON()
	if err != nil {
		return nil, err
	}
	err = writeFile("metadata", path, b)
	if err != nil {
		return nil, err
	}

	metadata.ToFiles(path)

	if request.Params.ListChangedFiles {
		cfol, err := github.GetChangedFiles(request.Version.PR)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch list of changed files: %s", err)
		}

		var fl []byte

		for _, v := range cfol {
			fl = append(fl, []byte(v+"\n")...)
		}

		// Create List with changed files
		if err := ioutil.WriteFile(filepath.Join(path, "changed_files"), fl, 0644); err != nil {
			return nil, fmt.Errorf("failed to write file list: %s", err)
		}
	}

	return &GetResponse{
		Version:  request.Version,
		Metadata: metadata,
	}, nil
}

func writeFile(name, path string, b []byte) error {
	if err := ioutil.WriteFile(filepath.Join(path, name+".json"), b, 0644); err != nil {
		return fmt.Errorf("failed to write %s: %s", name, err)
	}
	return nil
}

// GetParameters ...
type GetParameters struct {
	// SkipDownload will skip downloading the code to the volume, used with `put` steps
	SkipDownload bool `json:"skip_download"`
	// IntegrationTool defines the method of checking out the code (checkout [default], merge, rebase)
	IntegrationTool string `json:"integration_tool"`
	// GitDepth sets the number of commits to include in the clone (shallow clone)
	GitDepth int `json:"git_depth"`
	// ListChangedFiles generates a list of changed files in the `.git` directory
	ListChangedFiles bool `json:"list_changed_files"`
}

// GetRequest ...
type GetRequest struct {
	Source  Source        `json:"source"`
	Version Version       `json:"version"`
	Params  GetParameters `json:"params"`
}

// GetResponse ...
type GetResponse struct {
	Version  Version    `json:"version"`
	Metadata m.Metadata `json:"metadata,omitempty"`
}
