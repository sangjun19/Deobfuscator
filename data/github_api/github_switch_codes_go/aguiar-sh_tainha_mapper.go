// Repository: aguiar-sh/tainha
// File: internal/mapper/mapper.go

package mapper

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/aguiar-sh/tainha/internal/config"
	"github.com/aguiar-sh/tainha/internal/util"
)

func Map(route config.Route, response []byte) ([]byte, error) {
	var responseData interface{}
	if err := json.Unmarshal(response, &responseData); err != nil {
		log.Println("Error parsing JSON:", err)
		return nil, fmt.Errorf("failed to parse response body: %w", err)
	}

	var dataToProcess []map[string]interface{}
	switch v := responseData.(type) {
	case []interface{}:
		dataToProcess = make([]map[string]interface{}, len(v))
		for i, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				dataToProcess[i] = m
			} else {
				return nil, fmt.Errorf("invalid item in array at index %d", i)
			}
		}
	case map[string]interface{}:
		dataToProcess = []map[string]interface{}{v}
	default:
		return nil, fmt.Errorf("unsupported response type: %T", responseData)
	}

	var wg sync.WaitGroup
	errChan := make(chan error, len(dataToProcess)*len(route.Mapping))

	for i := range dataToProcess {
		for _, mapping := range route.Mapping {
			wg.Add(1)
			go func(item map[string]interface{}, mapping config.RouteMapping) {
				defer wg.Done()
				pathParams := util.ExtractPathParams(mapping.Path)

				for _, param := range pathParams {
					value, exists := item[param]
					if !exists {
						continue
					}

					valueStr := fmt.Sprintf("%v", value)
					path, protocol := util.PathProtocol(mapping.Service)

					fullPath := fmt.Sprintf("%s://%s", protocol, path)

					mappedURL := fmt.Sprintf("%s%s%s", fullPath, strings.ReplaceAll(mapping.Path, "{"+param+"}", ""), valueStr)

					log.Printf("Mapping: %s\n", mappedURL)

					resp, err := http.Get(mappedURL)
					if err != nil {
						errChan <- fmt.Errorf("error making request to %s: %v", mappedURL, err)
						return
					}
					defer resp.Body.Close()

					body, err := io.ReadAll(resp.Body)
					if err != nil {
						errChan <- fmt.Errorf("error reading response from %s: %v", mappedURL, err)
						return
					}

					var mappedData interface{}
					if err := json.Unmarshal(body, &mappedData); err != nil {
						errChan <- fmt.Errorf("error parsing JSON from %s: %v", mappedURL, err)
						return
					}

					item[mapping.Tag] = mappedData

					if mapping.RemoveKeyMapping {
						delete(item, param)
					}
				}
			}(dataToProcess[i], mapping)
		}
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			log.Println(err)
		}
	}

	finalResponse, err := json.Marshal(responseData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal final response: %w", err)
	}

	return finalResponse, nil
}
