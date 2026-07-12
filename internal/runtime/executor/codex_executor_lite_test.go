package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestCodexExecutorResponsesLiteHTTPRequests(t *testing.T) {
	const model = "test-codex-lite-header-precedence-http"
	reg := registry.GetGlobalRegistry()
	clientID := "test-codex-lite-header-precedence-http-client"
	reg.RegisterClient(clientID, "codex", []*registry.ModelInfo{{
		ID: model,
		Config: &registry.ModelConfig{OverrideHeader: map[string]string{
			codexResponsesLiteHeader: "model-override",
		}},
	}})
	t.Cleanup(func() { reg.UnregisterClient(clientID) })

	for _, stream := range []bool{false, true} {
		name := "execute"
		if stream {
			name = "execute stream"
		}
		t.Run(name, func(t *testing.T) {
			captured := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get(codexResponsesLiteHeader); got != "TRUE" {
					t.Errorf("%s = %q, want TRUE", codexResponsesLiteHeader, got)
				}
				body, _ := io.ReadAll(r.Body)
				captured <- body
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"output\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}\n\n"))
			}))
			defer server.Close()

			exec := NewCodexExecutor(&config.Config{})
			auth := &cliproxyauth.Auth{Attributes: map[string]string{
				"api_key":                            "sk-test",
				"base_url":                           server.URL,
				"header:" + codexResponsesLiteHeader: "auth-custom",
			}}
			req := cliproxyexecutor.Request{Model: model, Payload: []byte(`{"model":"gpt-5-codex","input":"hello","parallel_tool_calls":true,"tools":[{"type":"web_search"},{"type":"function","name":"lookup"}]}`)}
			opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Headers: http.Header{codexResponsesLiteHeader: []string{" TRUE "}}}

			if stream {
				result, err := exec.ExecuteStream(context.Background(), auth, req, opts)
				if err != nil {
					t.Fatalf("ExecuteStream() error = %v", err)
				}
				for chunk := range result.Chunks {
					if chunk.Err != nil {
						t.Fatalf("stream chunk error = %v", chunk.Err)
					}
				}
			} else if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			body := <-captured
			assertCodexLiteUpstreamPayload(t, body)
		})
	}
}

func assertCodexLiteUpstreamPayload(t *testing.T, body []byte) {
	t.Helper()
	if got := gjson.GetBytes(body, "reasoning.context").String(); got != "all_turns" {
		t.Fatalf("reasoning.context = %q, want all_turns; body=%s", got, body)
	}
	if parallel := gjson.GetBytes(body, "parallel_tool_calls"); !parallel.Exists() || parallel.Bool() {
		t.Fatalf("parallel_tool_calls = %s, want false; body=%s", parallel.Raw, body)
	}
	tools := gjson.GetBytes(body, "tools").Array()
	if len(tools) != 1 || tools[0].Get("type").String() != "function" {
		t.Fatalf("tools = %s, want only function tool", gjson.GetBytes(body, "tools").Raw)
	}
	for _, tool := range tools {
		if tool.Get("type").String() == "image_generation" {
			t.Fatalf("image_generation was injected: body=%s", body)
		}
	}
}

func TestForwardCodexResponsesLiteHeaderRemovesNonExecutionValue(t *testing.T) {
	headers := http.Header{codexResponsesLiteHeader: []string{"auth-or-model"}}
	forwardCodexResponsesLiteHeader(headers, nil)
	if got := headers.Get(codexResponsesLiteHeader); got != "" {
		t.Fatalf("%s = %q, want absent without execution header", codexResponsesLiteHeader, got)
	}
}
