package executor

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	internalcache "github.com/router-for-me/CLIProxyAPI/v7/internal/cache"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestBuildCodexWebsocketRequestBodyPreservesPreviousResponseID(t *testing.T) {
	body := []byte(`{"model":"gpt-5-codex","previous_response_id":"resp-1","input":[{"type":"message","id":"msg-1"}]}`)

	wsReqBody := buildCodexWebsocketRequestBody(body)

	if got := gjson.GetBytes(wsReqBody, "type").String(); got != "response.create" {
		t.Fatalf("type = %s, want response.create", got)
	}
	if got := gjson.GetBytes(wsReqBody, "previous_response_id").String(); got != "resp-1" {
		t.Fatalf("previous_response_id = %s, want resp-1", got)
	}
	if gjson.GetBytes(wsReqBody, "input.0.id").String() != "msg-1" {
		t.Fatalf("input item id mismatch")
	}
	if got := gjson.GetBytes(wsReqBody, "type").String(); got == "response.append" {
		t.Fatalf("unexpected websocket request type: %s", got)
	}
}

func TestBuildCodexWebsocketRequestBodySanitizesOverlongInputItemIDs(t *testing.T) {
	longReasoningItemID := "rs_" + strings.Repeat("a", 64)
	longCallItemID := strings.Repeat("grok-call-item-", 6)
	longOutputItemID := strings.Repeat("grok-output-item-", 6)
	body := []byte(`{"model":"gpt-5-codex","input":[{"type":"reasoning","id":"` + longReasoningItemID + `","encrypted_content":"gAAAA-encrypted","summary":[]},{"type":"function_call","id":"` + longCallItemID + `","call_id":"call-1","name":"lookup"},{"type":"function_call_output","id":"` + longOutputItemID + `","call_id":"call-1","output":"ok"},{"type":"message","id":"msg-1"}]}`)

	first := buildCodexWebsocketRequestBody(body)
	second := buildCodexWebsocketRequestBody(body)

	if input := gjson.GetBytes(first, "input").Array(); len(input) != 3 {
		t.Fatalf("input length = %d, want 3: %s", len(input), first)
	}
	if gotType := gjson.GetBytes(first, "input.0.type").String(); gotType != "function_call" {
		t.Fatalf("input.0.type = %q, want function_call: %s", gotType, first)
	}

	shortCallItemID := gjson.GetBytes(first, "input.0.id").String()
	shortOutputItemID := gjson.GetBytes(first, "input.1.id").String()
	if len([]rune(shortCallItemID)) > 64 || shortCallItemID == longCallItemID {
		t.Fatalf("input.0.id was not shortened to at most 64 characters: %q", shortCallItemID)
	}
	if len([]rune(shortOutputItemID)) > 64 || shortOutputItemID == longOutputItemID {
		t.Fatalf("input.1.id was not shortened to at most 64 characters: %q", shortOutputItemID)
	}
	if shortCallItemID == shortOutputItemID {
		t.Fatalf("distinct long IDs produced the same shortened ID: %q", shortCallItemID)
	}
	if got := gjson.GetBytes(second, "input.0.id").String(); got != shortCallItemID {
		t.Fatalf("input item ID shortening is not deterministic: first=%q second=%q", shortCallItemID, got)
	}
	if got := gjson.GetBytes(first, "input.0.call_id").String(); got != "call-1" {
		t.Fatalf("function call_id = %q, want call-1", got)
	}
	if got := gjson.GetBytes(first, "input.1.call_id").String(); got != "call-1" {
		t.Fatalf("function call output call_id = %q, want call-1", got)
	}
	if got := gjson.GetBytes(first, "input.2.id").String(); got != "msg-1" {
		t.Fatalf("valid input item ID changed: %q", got)
	}
}

func TestCodexWebsocketsExecuteRestoresClaudeAgentReasoningReplay(t *testing.T) {
	internalcache.ClearCodexReasoningReplayCache()
	t.Cleanup(internalcache.ClearCodexReasoningReplayCache)

	encryptedContent := validCodexReasoningEncryptedContentForTestSeed(31)
	cacheCodexReasoningReplayFromCompleted(codexReasoningReplayScope{
		modelName:  "gpt-5.4",
		sessionKey: "claude:ws-replay-session:agent:agent-a",
	}, []byte(`{"response":{"output":[`+
		`{"type":"reasoning","summary":[],"content":null,"encrypted_content":"`+encryptedContent+`"},`+
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"previous answer"}]}`+
		`]}}`))

	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	capturedPayload := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Fatalf("upgrade websocket: %v", errUpgrade)
		}
		defer func() { _ = conn.Close() }()

		_, payload, errRead := conn.ReadMessage()
		if errRead != nil {
			t.Fatalf("read upstream websocket message: %v", errRead)
		}
		capturedPayload <- bytes.Clone(payload)
		completed := []byte(`{"type":"response.completed","response":{"id":"resp-ws-replay","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"next answer"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Fatalf("write completed websocket message: %v", errWrite)
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model: "gpt-5.4",
		Payload: []byte(`{
			"model":"gpt-5.4",
			"messages":[
				{"role":"user","content":"first"},
				{"role":"assistant","content":"previous answer"},
				{"role":"user","content":"next"}
			]
		}`),
	}
	headers := http.Header{}
	headers.Set("X-Claude-Code-Session-Id", "ws-replay-session")
	headers.Set("X-Claude-Code-Agent-Id", "agent-a")
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Headers: headers}

	if _, errExecute := exec.Execute(context.Background(), auth, req, opts); errExecute != nil {
		t.Fatalf("Execute() error = %v", errExecute)
	}

	select {
	case payload := <-capturedPayload:
		input := gjson.GetBytes(payload, "input").Array()
		if len(input) != 4 {
			t.Fatalf("upstream input length = %d, want 4; payload=%s", len(input), payload)
		}
		if input[1].Get("type").String() != "reasoning" || input[1].Get("encrypted_content").String() != encryptedContent {
			t.Fatalf("websocket reasoning replay missing before assistant message: %s", payload)
		}
		if input[2].Get("role").String() != "assistant" {
			t.Fatalf("input.2.role = %q, want assistant; payload=%s", input[2].Get("role").String(), payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream websocket payload")
	}
}

func TestClearCodexReasoningReplayOnWebsocketInvalidSignature(t *testing.T) {
	internalcache.ClearCodexReasoningReplayCache()
	t.Cleanup(internalcache.ClearCodexReasoningReplayCache)

	scope := codexReasoningReplayScope{modelName: "gpt-5.4", sessionKey: "claude:ws-invalid:agent:main"}
	encryptedContent := validCodexReasoningEncryptedContentForTestSeed(32)
	if !internalcache.CacheCodexReasoningReplayItem(scope.modelName, scope.sessionKey, []byte(`{"type":"reasoning","summary":[],"content":null,"encrypted_content":"`+encryptedContent+`"}`)) {
		t.Fatal("failed to seed websocket replay cache")
	}
	payload := []byte(`{"type":"error","status":400,"body":{"error":{"message":"Invalid signature in thinking block","type":"invalid_request_error","code":"invalid_request_error"}}}`)
	if errClear := clearCodexReasoningReplayOnWebsocketError(context.Background(), scope, payload); errClear != nil {
		t.Fatalf("clear websocket replay error: %v", errClear)
	}
	if _, ok := internalcache.GetCodexReasoningReplayItem(scope.modelName, scope.sessionKey); ok {
		t.Fatal("websocket invalid signature did not clear replay state")
	}
}

func TestCodexWebsocketsExecuteResponsesLiteDoesNotInjectImageGenerationTool(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	capturedPayload := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer func() { _ = conn.Close() }()

		_, payload, errRead := conn.ReadMessage()
		if errRead != nil {
			t.Fatalf("read upstream websocket message: %v", errRead)
		}
		capturedPayload <- bytes.Clone(payload)

		completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Fatalf("write completed websocket message: %v", errWrite)
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":   "sk-test",
			"base_url":  server.URL,
			"plan_type": "pro",
		},
	}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.6-sol",
		Payload: []byte(`{"model":"gpt-5.6-sol","input":[{"type":"additional_tools","role":"developer","tools":[{"type":"custom","name":"exec"}]},{"role":"user","content":"hello"}],"parallel_tool_calls":true,"client_metadata":{"ws_request_header_x_openai_internal_codex_responses_lite":"true"}}`),
	}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("codex")}

	if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	select {
	case payload := <-capturedPayload:
		if tools := gjson.GetBytes(payload, "tools"); tools.Exists() {
			t.Fatalf("unexpected tools in responses-lite upstream payload: %s", tools.Raw)
		}
		if got := gjson.GetBytes(payload, "input.0.type").String(); got != "additional_tools" {
			t.Fatalf("input.0.type = %q, want additional_tools; payload=%s", got, payload)
		}
		if got := gjson.GetBytes(payload, "client_metadata.ws_request_header_x_openai_internal_codex_responses_lite").String(); got != "true" {
			t.Fatalf("responses-lite metadata = %q, want true; payload=%s", got, payload)
		}
		parallelToolCalls := gjson.GetBytes(payload, "parallel_tool_calls")
		if !parallelToolCalls.Exists() || parallelToolCalls.Bool() {
			t.Fatalf("responses-lite parallel_tool_calls should be false: %s", payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream websocket payload")
	}
}

func TestCodexWebsocketsExecuteStreamResponsesLiteForcesParallelToolCallsFalse(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	capturedPayload := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Errorf("upgrade websocket: %v", errUpgrade)
			return
		}
		defer func() { _ = conn.Close() }()

		_, payload, errRead := conn.ReadMessage()
		if errRead != nil {
			t.Errorf("read upstream websocket message: %v", errRead)
			return
		}
		capturedPayload <- bytes.Clone(payload)

		completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Errorf("write completed websocket message: %v", errWrite)
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":   "sk-test",
			"base_url":  server.URL,
			"plan_type": "pro",
		},
	}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.6-luna",
		Payload: []byte(`{"model":"gpt-5.6-luna","input":[{"type":"additional_tools","role":"developer","tools":[{"type":"custom","name":"exec"}]},{"role":"user","content":"hello"}],"parallel_tool_calls":true,"client_metadata":{"ws_request_header_x_openai_internal_codex_responses_lite":"true"}}`),
	}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("codex")}

	result, errExecute := exec.ExecuteStream(context.Background(), auth, req, opts)
	if errExecute != nil {
		t.Fatalf("ExecuteStream() error = %v", errExecute)
	}
	streamComplete := false
	for !streamComplete {
		select {
		case chunk, ok := <-result.Chunks:
			if !ok {
				streamComplete = true
				continue
			}
			if chunk.Err != nil {
				t.Fatalf("stream chunk error = %v", chunk.Err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for websocket stream completion")
		}
	}

	select {
	case payload := <-capturedPayload:
		parallelToolCalls := gjson.GetBytes(payload, "parallel_tool_calls")
		if !parallelToolCalls.Exists() || parallelToolCalls.Bool() {
			t.Fatalf("responses-lite parallel_tool_calls should be false: %s", payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream websocket payload")
	}
}

func TestCodexWebsocketsExecutePreservesPreviousResponseIDUpstream(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	capturedPayload := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("request path = %s, want /responses", r.URL.Path)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer func() { _ = conn.Close() }()

		msgType, payload, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read upstream websocket message: %v", err)
		}
		if msgType != websocket.TextMessage {
			t.Fatalf("message type = %d, want text", msgType)
		}
		capturedPayload <- bytes.Clone(payload)

		completed := []byte(`{"type":"response.completed","response":{"id":"resp-2","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Fatalf("write completed websocket message: %v", errWrite)
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"gpt-5-codex","previous_response_id":"resp-1","input":[{"type":"message","id":"msg-1"}]}`),
	}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("codex")}

	if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	select {
	case payload := <-capturedPayload:
		if got := gjson.GetBytes(payload, "type").String(); got != "response.create" {
			t.Fatalf("upstream type = %s, want response.create; payload=%s", got, payload)
		}
		if got := gjson.GetBytes(payload, "previous_response_id").String(); got != "resp-1" {
			t.Fatalf("upstream previous_response_id = %s, want resp-1; payload=%s", got, payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream websocket payload")
	}
}

func TestCodexWebsocketsExecuteStreamUpgradeRequiredReturnsWithoutLockingSession(t *testing.T) {
	upgradeAttempts := make(chan struct{}, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			t.Errorf("unexpected HTTP fallback request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		upgradeAttempts <- struct{}{}
		w.WriteHeader(http.StatusUpgradeRequired)
		_, _ = w.Write([]byte(`{"error":{"message":"websocket unavailable"}}`))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	const executionSessionID = "ws-upgrade-required-session"
	t.Cleanup(func() { exec.CloseExecutionSession(executionSessionID) })
	auth := &cliproxyauth.Auth{
		ID:       "codex-test",
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":  "sk-test",
			"base_url": server.URL,
		},
	}
	opts := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FromString("openai-response"),
		ResponseFormat: sdktranslator.FromString("openai-response"),
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: executionSessionID,
		},
	}
	ctx := cliproxyexecutor.WithDownstreamWebsocket(context.Background())

	execute := func(payload string) {
		t.Helper()
		done := make(chan error, 1)
		go func() {
			_, errExecute := exec.ExecuteStream(ctx, auth, cliproxyexecutor.Request{
				Model:   "gpt-5.4",
				Payload: []byte(payload),
			}, opts)
			done <- errExecute
		}()

		select {
		case errExecute := <-done:
			if errExecute == nil {
				t.Fatal("upgrade-required error = nil")
			}
			statusErr, ok := errExecute.(interface{ StatusCode() int })
			if !ok || statusErr.StatusCode() != http.StatusUpgradeRequired {
				t.Fatalf("upgrade-required error = %T %v, want status 426", errExecute, errExecute)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for upgrade-required error; execution session may still be locked")
		}
	}

	execute(`{"model":"gpt-5.4","generate":false,"input":[]}`)
	execute(`{"model":"gpt-5.4","previous_response_id":"resp-1","input":[{"type":"message","id":"msg-2"}]}`)

	if got := len(upgradeAttempts); got != 2 {
		t.Fatalf("websocket upgrade attempts = %d, want 2", got)
	}
}

func TestCodexWebsocketsExecuteStreamHandshakeErrorReturnsWithoutLockingSession(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"unauthorized"}}`))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	const executionSessionID = "ws-handshake-error-session"
	t.Cleanup(func() { exec.CloseExecutionSession(executionSessionID) })
	auth := &cliproxyauth.Auth{
		ID:       "codex-test",
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":  "sk-test",
			"base_url": server.URL,
		},
	}
	opts := cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: executionSessionID,
		},
	}

	for i := 0; i < 2; i++ {
		done := make(chan error, 1)
		go func() {
			_, errExecute := exec.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
				Model:   "gpt-5.4",
				Payload: []byte(`{"model":"gpt-5.4","input":[{"type":"message","id":"msg-1"}]}`),
			}, opts)
			done <- errExecute
		}()
		select {
		case errExecute := <-done:
			statusErr, ok := errExecute.(interface{ StatusCode() int })
			if !ok || statusErr.StatusCode() != http.StatusUnauthorized {
				t.Fatalf("attempt %d error = %T %v, want status 401", i+1, errExecute, errExecute)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("attempt %d timed out; execution session remained locked", i+1)
		}
	}
}

func TestExistingWebsocketSessionConnRequiresMatchingHealthyConnection(t *testing.T) {
	conn := &websocket.Conn{}
	closer := newWebsocketConnectionCloser(conn)
	sess := &codexWebsocketSession{
		conn:       conn,
		connCloser: closer,
		authID:     "auth-a",
		wsURL:      "ws://example.test/responses",
	}
	sess.resetUpstreamDisconnectError(conn)
	if gotConn, gotCloser := existingWebsocketSessionConn(sess, "auth-a", "ws://example.test/responses"); gotConn != conn || gotCloser != closer {
		t.Fatal("matching healthy websocket session was not reusable")
	}
	if got, _ := existingWebsocketSessionConn(sess, "auth-b", "ws://example.test/responses"); got != nil {
		t.Fatal("websocket session matched a different auth")
	}
	if got, _ := existingWebsocketSessionConn(sess, "auth-a", "ws://other.test/responses"); got != nil {
		t.Fatal("websocket session matched a different URL")
	}
	sess.setUpstreamDisconnectError(conn, errors.New("upstream disconnected"))
	if got, _ := existingWebsocketSessionConn(sess, "auth-a", "ws://example.test/responses"); got != nil {
		t.Fatal("disconnected websocket session remained reusable")
	}
}

func TestCodexAutoExecutorRequiredUpstreamWebsocketRejectsHTTPFallback(t *testing.T) {
	exec := NewCodexAutoExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{
		ID:       "codex-http-only",
		Provider: "codex",
		Attributes: map[string]string{
			"api_key": "sk-test",
		},
	}
	ctx := cliproxyexecutor.WithRequiredUpstreamWebsocket(
		cliproxyexecutor.WithDownstreamWebsocket(context.Background()),
	)
	_, errExecute := exec.ExecuteStream(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(`{"model":"gpt-5.4","previous_response_id":"resp-1","input":[{"type":"message","id":"msg-2"}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")})
	if errExecute == nil {
		t.Fatal("ExecuteStream() error = nil, want replay-required error")
	}
	statusErr, ok := errExecute.(interface{ StatusCode() int })
	if !ok || statusErr.StatusCode() != http.StatusUpgradeRequired {
		t.Fatalf("ExecuteStream() error = %T %v, want status 426", errExecute, errExecute)
	}
	if got := gjson.Get(errExecute.Error(), "error.code").String(); got != "upstream_http_replay_required" {
		t.Fatalf("ExecuteStream() error code = %q, want upstream_http_replay_required", got)
	}
	requestScoped, ok := errExecute.(cliproxyexecutor.RequestScopedError)
	if !ok || !requestScoped.IsRequestScoped() {
		t.Fatalf("ExecuteStream() error = %T, want request-scoped replay signal", errExecute)
	}
}

func TestCodexWebsocketsExecuteStreamPassesThroughUpstreamWebsocketPayloadForDownstreamWebsocket(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	capturedPayload := make(chan []byte, 1)
	delta := []byte(`{"type":"response.output_text.delta","delta":"hello"}`)
	completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()

		_, payload, errRead := conn.ReadMessage()
		if errRead != nil {
			t.Errorf("read upstream websocket message: %v", errRead)
			return
		}
		capturedPayload <- bytes.Clone(payload)
		if errWrite := conn.WriteMessage(websocket.TextMessage, delta); errWrite != nil {
			t.Errorf("write delta websocket message: %v", errWrite)
			return
		}
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Errorf("write completed websocket message: %v", errWrite)
			return
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"prolite/gpt-5-codex","input":[{"type":"additional_tools","role":"developer","tools":[{"type":"custom","name":"exec"}]},{"type":"message","role":"user","content":"hello"}],"parallel_tool_calls":true}`),
	}
	opts := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FromString("openai-response"),
		ResponseFormat: sdktranslator.FromString("openai-response"),
	}
	ctx := cliproxyexecutor.WithDownstreamWebsocket(context.Background())

	result, err := exec.ExecuteStream(ctx, auth, req, opts)
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}

	select {
	case chunk, ok := <-result.Chunks:
		if !ok {
			t.Fatal("stream closed before first chunk")
		}
		if chunk.Err != nil {
			t.Fatalf("first chunk error = %v", chunk.Err)
		}
		if !bytes.Equal(bytes.TrimSpace(chunk.Payload), delta) {
			t.Fatalf("first chunk = %q, want raw upstream websocket payload %q", chunk.Payload, delta)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for first stream chunk")
	}

	select {
	case payload := <-capturedPayload:
		if got := gjson.GetBytes(payload, "model").String(); got != "gpt-5-codex" {
			t.Fatalf("upstream model = %s, want gpt-5-codex; payload=%s", got, payload)
		}
		parallelToolCalls := gjson.GetBytes(payload, "parallel_tool_calls")
		if !parallelToolCalls.Exists() || !parallelToolCalls.Bool() {
			t.Fatalf("non-lite parallel_tool_calls should be preserved: %s", payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream websocket payload")
	}
}

func TestCodexWebsocketsResponsesLiteExecutionHeaders(t *testing.T) {
	const model = "test-codex-lite-header-precedence-ws"
	reg := registry.GetGlobalRegistry()
	clientID := "test-codex-lite-header-precedence-ws-client"
	reg.RegisterClient(clientID, "codex", []*registry.ModelInfo{{
		ID: model,
		Config: &registry.ModelConfig{OverrideHeader: map[string]string{
			codexResponsesLiteHeader: "model-override",
		}},
	}})
	t.Cleanup(func() { reg.UnregisterClient(clientID) })

	tests := []struct {
		name          string
		stream        bool
		headerValue   string
		payload       string
		wantImageTool bool
		wantTools     string
		wantReasoning string
		wantChoice    string
	}{
		{
			name:          "execute lite true without gin",
			headerValue:   " TRUE ",
			payload:       `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}],"parallel_tool_calls":true,"tools":[{"type":" image_generation "},{"type":"function","name":"lookup"},{"type":"custom","name":"custom"},{"type":"tool_search"},{"type":"local_shell"},{"type":"apply_patch"},{"type":"namespace"},{"type":"future_tool"}],"tool_choice":{"type":"IMAGE_GENERATION"}}`,
			wantTools:     `[{"type":"function","name":"lookup"},{"type":"custom","name":"custom"},{"type":"tool_search"},{"type":"local_shell"},{"type":"apply_patch"},{"type":"namespace"},{"type":"future_tool"}]`,
			wantReasoning: `{"context":"all_turns"}`,
		},
		{
			name:          "execute stream lite true without gin",
			stream:        true,
			headerValue:   "true",
			payload:       `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}],"reasoning":{"effort":"high","summary":"detailed","context":"last_turn"},"tools":[{"type":"web_search"},{"type":"WEB_SEARCH_PREVIEW"},{"type":" file_search "},{"type":"code_interpreter"},{"type":"computer_use"},{"type":"computer_use_preview"},{"type":"function","name":"lookup"}],"tool_choice":"auto"}`,
			wantTools:     `[{"type":"function","name":"lookup"}]`,
			wantReasoning: `{"summary":"detailed","context":"all_turns"}`,
			wantChoice:    `"auto"`,
		},
		{
			name:          "lite false retains normal injection",
			headerValue:   "false",
			payload:       `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`,
			wantImageTool: true,
		},
		{
			name:          "lite invalid retains normal injection",
			headerValue:   "not-a-bool",
			payload:       `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`,
			wantImageTool: true,
		},
		{
			name:          "lite preserves existing tools",
			stream:        true,
			headerValue:   "true",
			payload:       `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}],"tools":[{"type":"function","name":"lookup","description":"find it","parameters":{"type":"object","properties":{"id":{"type":"string"}}}}]}`,
			wantTools:     `[{"type":"function","name":"lookup","description":"find it","parameters":{"type":"object","properties":{"id":{"type":"string"}}}}]`,
			wantReasoning: `{"context":"all_turns"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
			type upstreamRequest struct {
				liteHeader string
				payload    []byte
			}
			captured := make(chan upstreamRequest, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := upgrader.Upgrade(w, r, nil)
				if err != nil {
					t.Errorf("upgrade websocket: %v", err)
					return
				}
				defer func() { _ = conn.Close() }()

				_, payload, errRead := conn.ReadMessage()
				if errRead != nil {
					t.Errorf("read upstream websocket message: %v", errRead)
					return
				}
				captured <- upstreamRequest{liteHeader: r.Header.Get(codexResponsesLiteHeader), payload: bytes.Clone(payload)}

				completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
				if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
					t.Errorf("write completed websocket message: %v", errWrite)
				}
			}))
			defer server.Close()

			exec := NewCodexWebsocketsExecutor(&config.Config{})
			auth := &cliproxyauth.Auth{Attributes: map[string]string{
				"api_key":                            "sk-test",
				"base_url":                           server.URL,
				"header:" + codexResponsesLiteHeader: "auth-custom",
			}}
			req := cliproxyexecutor.Request{Model: model, Payload: []byte(tt.payload)}
			opts := cliproxyexecutor.Options{
				SourceFormat: sdktranslator.FromString("openai-response"),
				Headers:      http.Header{strings.ToLower(codexResponsesLiteHeader): []string{tt.headerValue}},
			}

			if tt.stream {
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

			select {
			case got := <-captured:
				decision := resolveCodexResponsesLite([]byte(tt.payload), opts.Headers, model)
				wantHeader := decision.headerValue
				if wantHeader == "" && decision.enabled() {
					wantHeader = "true"
				} else if wantHeader == "" && decision.state == codexResponsesLiteExplicitFalse {
					wantHeader = "false"
				}
				if got.liteHeader != wantHeader {
					t.Fatalf("upstream %s = %q, want %q", codexResponsesLiteHeader, got.liteHeader, wantHeader)
				}
				hasImageTool := false
				for _, tool := range gjson.GetBytes(got.payload, "tools").Array() {
					if tool.Get("type").String() == "image_generation" {
						hasImageTool = true
					}
				}
				if hasImageTool != tt.wantImageTool {
					t.Fatalf("image_generation present = %v, want %v; payload=%s", hasImageTool, tt.wantImageTool, got.payload)
				}
				if tt.wantTools != "" && gjson.GetBytes(got.payload, "tools").Raw != tt.wantTools {
					t.Fatalf("tools = %s, want semantically unchanged %s", gjson.GetBytes(got.payload, "tools").Raw, tt.wantTools)
				}
				if gotReasoning := gjson.GetBytes(got.payload, "reasoning").Raw; gotReasoning != tt.wantReasoning {
					t.Fatalf("reasoning = %s, want %s; payload=%s", gotReasoning, tt.wantReasoning, got.payload)
				}
				if decision.enabled() {
					if parallel := gjson.GetBytes(got.payload, "parallel_tool_calls"); !parallel.Exists() || parallel.Bool() {
						t.Fatalf("parallel_tool_calls = %s, want false; payload=%s", parallel.Raw, got.payload)
					}
				}
				if gotChoice := gjson.GetBytes(got.payload, "tool_choice"); tt.wantChoice == "" && gotChoice.Exists() {
					t.Fatalf("tool_choice = %s, want absent; payload=%s", gotChoice.Raw, got.payload)
				} else if tt.wantChoice != "" && gotChoice.Raw != tt.wantChoice {
					t.Fatalf("tool_choice = %s, want %s; payload=%s", gotChoice.Raw, tt.wantChoice, got.payload)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for upstream websocket request")
			}
		})
	}
}

func TestCodexWebsocketsResponsesLiteInferredFromCatalog(t *testing.T) {
	tests := []struct {
		name          string
		stream        bool
		payload       string
		headers       http.Header
		wantLite      bool
		wantHeader    string
		wantImageTool bool
	}{
		{name: "execute without tools", payload: `{"model":"gpt-5.6-sol","input":"hello","parallel_tool_calls":true}`, wantLite: true, wantHeader: "true"},
		{name: "execute stream with hosted tools filtered", stream: true, payload: `{"model":"gpt-5.6-sol","input":"hello","tools":[{"type":"web_search"},{"type":"image_generation"}]}`, wantLite: true, wantHeader: "true"},
		{name: "explicit false header", payload: `{"model":"gpt-5.6-sol","input":"hello"}`, headers: http.Header{codexResponsesLiteHeader: []string{"false"}}, wantHeader: "false", wantImageTool: true},
		{name: "explicit false metadata stream", stream: true, payload: `{"model":"gpt-5.6-sol","client_metadata":{"ws_request_header_x_openai_internal_codex_responses_lite":false},"input":"hello"}`, wantHeader: "false", wantImageTool: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
			type upstreamRequest struct {
				liteHeader string
				payload    []byte
			}
			captured := make(chan upstreamRequest, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := upgrader.Upgrade(w, r, nil)
				if err != nil {
					t.Errorf("upgrade websocket: %v", err)
					return
				}
				defer func() { _ = conn.Close() }()

				_, payload, errRead := conn.ReadMessage()
				if errRead != nil {
					t.Errorf("read upstream websocket message: %v", errRead)
					return
				}
				captured <- upstreamRequest{liteHeader: r.Header.Get(codexResponsesLiteHeader), payload: bytes.Clone(payload)}
				completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
				if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
					t.Errorf("write completed websocket message: %v", errWrite)
				}
			}))
			defer server.Close()

			exec := NewCodexWebsocketsExecutor(&config.Config{})
			auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
			req := cliproxyexecutor.Request{Model: "gpt-5.6-sol", Payload: []byte(tt.payload)}
			opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Headers: tt.headers}

			if tt.stream {
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

			select {
			case got := <-captured:
				if got.liteHeader != tt.wantHeader {
					t.Fatalf("upstream %s = %q, want %q", codexResponsesLiteHeader, got.liteHeader, tt.wantHeader)
				}
				if tt.wantLite {
					if contextValue := gjson.GetBytes(got.payload, "reasoning.context").String(); contextValue != "all_turns" {
						t.Fatalf("reasoning.context = %q, want all_turns; payload=%s", contextValue, got.payload)
					}
					if parallel := gjson.GetBytes(got.payload, "parallel_tool_calls"); !parallel.Exists() || parallel.Bool() {
						t.Fatalf("parallel_tool_calls = %s, want false; payload=%s", parallel.Raw, got.payload)
					}
					if tools := gjson.GetBytes(got.payload, "tools"); tools.Exists() && len(tools.Array()) != 0 {
						t.Fatalf("tools = %s, want missing or empty after Lite normalization", tools.Raw)
					}
				}
				hasImageTool := false
				for _, tool := range gjson.GetBytes(got.payload, "tools").Array() {
					if tool.Get("type").String() == "image_generation" {
						hasImageTool = true
					}
				}
				if hasImageTool != tt.wantImageTool {
					t.Fatalf("image_generation present = %v, want %v; payload=%s", hasImageTool, tt.wantImageTool, got.payload)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for upstream websocket request")
			}
		})
	}
}

func TestNormalizeCodexResponsesLiteRequest(t *testing.T) {
	tests := []struct {
		name               string
		body               string
		wantReasoning      string
		wantTools          string
		wantToolChoice     string
		wantToolChoiceGone bool
	}{
		{
			name:          "adds missing reasoning and parallel tool calls",
			body:          `{"model":"gpt-5-codex"}`,
			wantReasoning: `{"context":"all_turns"}`,
		},
		{
			name:          "preserves existing effort and summary",
			body:          `{"reasoning":{"effort":"high","summary":"detailed"},"parallel_tool_calls":true}`,
			wantReasoning: `{"effort":"high","summary":"detailed","context":"all_turns"}`,
		},
		{
			name:          "replaces conflicting context",
			body:          `{"reasoning":{"context":"last_turn"}}`,
			wantReasoning: `{"context":"all_turns"}`,
		},
		{
			name:               "filters hosted tools and removed hosted choice",
			body:               `{"tools":[{"type":" image_generation ","quality":"high"},{"type":"WEB_SEARCH"},{"type":"web_search_preview"},{"type":"file_search"},{"type":"code_interpreter"},{"type":"computer_use"},{"type":"computer_use_preview"},{"type":"function","name":"f"},{"type":"custom","name":"c"},{"type":"tool_search"},{"type":"local_shell"},{"type":"apply_patch"},{"type":"namespace"},{"type":"unknown","extra":{"x":1}}],"tool_choice":{"type":" web_search_preview "}}`,
			wantReasoning:      `{"context":"all_turns"}`,
			wantTools:          `[{"type":"function","name":"f"},{"type":"custom","name":"c"},{"type":"tool_search"},{"type":"local_shell"},{"type":"apply_patch"},{"type":"namespace"},{"type":"unknown","extra":{"x":1}}]`,
			wantToolChoiceGone: true,
		},
		{
			name:           "preserves generic and client tool choices",
			body:           `{"tools":[{"type":"function","name":"f"}],"tool_choice":{"type":"function","name":"f"}}`,
			wantReasoning:  `{"context":"all_turns"}`,
			wantTools:      `[{"type":"function","name":"f"}]`,
			wantToolChoice: `{"type":"function","name":"f"}`,
		},
		{
			name:           "preserves generic none choice",
			body:           `{"tool_choice":"none"}`,
			wantReasoning:  `{"context":"all_turns"}`,
			wantToolChoice: `"none"`,
		},
		{
			name:           "filters mixed allowed tools and preserves fields",
			body:           `{"tools":[{"type":"function","name":"f"}],"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"web_search"},{"type":"function","name":"f"},{"type":"future_tool","name":"x"}]}}`,
			wantReasoning:  `{"context":"all_turns"}`,
			wantTools:      `[{"type":"function","name":"f"}]`,
			wantToolChoice: `{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"f"},{"type":"future_tool","name":"x"}]}`,
		},
		{
			name:               "removes empty allowed tools choice",
			body:               `{"tools":[{"type":"function","name":"f"}],"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"web_search"},{"type":"image_generation"}]}}`,
			wantReasoning:      `{"context":"all_turns"}`,
			wantTools:          `[{"type":"function","name":"f"}]`,
			wantToolChoiceGone: true,
		},
		{
			name:               "removes string required when all tools removed",
			body:               `{"tools":[{"type":"web_search"}],"tool_choice":"required"}`,
			wantReasoning:      `{"context":"all_turns"}`,
			wantTools:          `[]`,
			wantToolChoiceGone: true,
		},
		{
			name:           "preserves object required when tools remain",
			body:           `{"tools":[{"type":"image_generation"},{"type":"function","name":"f"}],"tool_choice":{"type":"required","extra":"keep"}}`,
			wantReasoning:  `{"context":"all_turns"}`,
			wantTools:      `[{"type":"function","name":"f"}]`,
			wantToolChoice: `{"type":"required","extra":"keep"}`,
		},
	}

	decision := codexResponsesLiteDecision{state: codexResponsesLiteExplicitTrue}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeCodexResponsesLiteRequest([]byte(tt.body), decision)
			if gotReasoning := gjson.GetBytes(got, "reasoning").Raw; gotReasoning != tt.wantReasoning {
				t.Fatalf("reasoning = %s, want %s; body=%s", gotReasoning, tt.wantReasoning, got)
			}
			parallel := gjson.GetBytes(got, "parallel_tool_calls")
			if !parallel.Exists() || parallel.Bool() {
				t.Fatalf("parallel_tool_calls = %s, want false; body=%s", parallel.Raw, got)
			}
			if tt.wantTools != "" && gjson.GetBytes(got, "tools").Raw != tt.wantTools {
				t.Fatalf("tools = %s, want %s; body=%s", gjson.GetBytes(got, "tools").Raw, tt.wantTools, got)
			}
			choice := gjson.GetBytes(got, "tool_choice")
			if tt.wantToolChoiceGone && choice.Exists() {
				t.Fatalf("tool_choice = %s, want absent; body=%s", choice.Raw, got)
			}
			if tt.wantToolChoice != "" && choice.Raw != tt.wantToolChoice {
				t.Fatalf("tool_choice = %s, want %s; body=%s", choice.Raw, tt.wantToolChoice, got)
			}
		})
	}

	nonLite := []byte(`{"reasoning":{"context":"last_turn"},"parallel_tool_calls":true,"tools":[{"type":"web_search"}],"tool_choice":"web_search"}`)
	if got := normalizeCodexResponsesLiteRequest(nonLite, codexResponsesLiteDecision{}); !bytes.Equal(got, nonLite) {
		t.Fatalf("non-Lite body changed: got=%s want=%s", got, nonLite)
	}
}

func TestCodexWebsocketHTTPFallbackRequestPreservesFormatsAndClonesPayload(t *testing.T) {
	prepared := []byte(`{"model":"gpt-5-codex","input":"prepared"}`)
	original := []byte(`{"messages":[{"role":"user","content":"original"}]}`)
	existingOriginal := []byte(`{"messages":[{"role":"user","content":"existing"}]}`)

	tests := []struct {
		name           string
		opts           cliproxyexecutor.Options
		wantOriginal   []byte
		responseFormat sdktranslator.Format
	}{
		{
			name:         "populates original request and preserves unset response format",
			opts:         cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude},
			wantOriginal: original,
		},
		{
			name: "preserves explicit response format and existing original request",
			opts: cliproxyexecutor.Options{
				SourceFormat:    sdktranslator.FormatClaude,
				ResponseFormat:  sdktranslator.FormatOpenAIResponse,
				OriginalRequest: existingOriginal,
			},
			wantOriginal:   existingOriginal,
			responseFormat: sdktranslator.FormatOpenAIResponse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: original}
			fallbackReq, fallbackOpts := codexWebsocketHTTPFallbackRequest(req, tt.opts, prepared)

			if !bytes.Equal(fallbackReq.Payload, prepared) {
				t.Fatalf("fallback payload = %s, want %s", fallbackReq.Payload, prepared)
			}
			if len(prepared) > 0 && &fallbackReq.Payload[0] == &prepared[0] {
				t.Fatal("fallback payload aliases prepared payload")
			}
			if fallbackOpts.SourceFormat.String() != tt.opts.SourceFormat.String() {
				t.Fatalf("SourceFormat = %s, want %s", fallbackOpts.SourceFormat.String(), tt.opts.SourceFormat.String())
			}
			if fallbackOpts.ResponseFormat.String() != tt.responseFormat.String() {
				t.Fatalf("ResponseFormat = %s, want %s", fallbackOpts.ResponseFormat.String(), tt.responseFormat.String())
			}
			if !bytes.Equal(fallbackOpts.OriginalRequest, tt.wantOriginal) {
				t.Fatalf("OriginalRequest = %s, want %s", fallbackOpts.OriginalRequest, tt.wantOriginal)
			}
			if len(tt.opts.OriginalRequest) == 0 && len(original) > 0 && &fallbackOpts.OriginalRequest[0] == &original[0] {
				t.Fatal("populated OriginalRequest aliases request payload")
			}
		})
	}
}

func TestCodexWebsocketsExecuteResponsesLite426FallbackUsesNormalizedHTTPPayload(t *testing.T) {
	tests := []struct {
		name         string
		sourceFormat sdktranslator.Format
		payload      string
	}{
		{
			name:         "openai response",
			sourceFormat: sdktranslator.FormatOpenAIResponse,
			payload:      `{"model":"gpt-5-codex","input":"hello","parallel_tool_calls":true,"tools":[{"type":"web_search"},{"type":"function","name":"lookup"}]}`,
		},
		{
			name:         "claude",
			sourceFormat: sdktranslator.FormatClaude,
			payload:      `{"model":"gpt-5-codex","max_tokens":128,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","description":"lookup","input_schema":{"type":"object","properties":{}}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			captured := make(chan []byte, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
					w.WriteHeader(http.StatusUpgradeRequired)
					return
				}
				if got := r.Header.Get(codexResponsesLiteHeader); got != "true" {
					t.Errorf("%s = %q, want true", codexResponsesLiteHeader, got)
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read fallback body: %v", err)
					return
				}
				captured <- body
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"output\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}\n\n"))
			}))
			defer server.Close()

			exec := NewCodexWebsocketsExecutor(&config.Config{})
			auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
			req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(tt.payload)}
			opts := cliproxyexecutor.Options{
				SourceFormat:   tt.sourceFormat,
				ResponseFormat: tt.sourceFormat,
				Headers:        http.Header{codexResponsesLiteHeader: []string{"true"}},
			}

			if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			select {
			case body := <-captured:
				if !gjson.ValidBytes(body) {
					t.Fatalf("fallback body is invalid JSON: %s", body)
				}
				assertCodexLiteUpstreamPayload(t, body)
				if gjson.GetBytes(body, "messages").Exists() {
					t.Fatalf("fallback body retained source messages instead of normalized Codex input: %s", body)
				}
				if input := gjson.GetBytes(body, "input"); !input.Exists() {
					t.Fatalf("fallback body missing Codex input: %s", body)
				}
				if !strings.Contains(gjson.GetBytes(body, "input").Raw, "hello") {
					t.Fatalf("fallback Codex input lost user content: %s", body)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for HTTP fallback request")
			}
		})
	}
}

func TestCodexWebsocketsExecute426FallbackPreservesImplicitClaudeResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			w.WriteHeader(http.StatusUpgradeRequired)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-5-codex\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello from fallback\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n"))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"gpt-5-codex","max_tokens":128,"messages":[{"role":"user","content":"hello"}]}`),
	}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude}

	resp, err := exec.Execute(context.Background(), auth, req, opts)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got := gjson.GetBytes(resp.Payload, "type").String(); got != "message" {
		t.Fatalf("response type = %q, want Claude message; payload=%s", got, resp.Payload)
	}
	if got := gjson.GetBytes(resp.Payload, "content.0.text").String(); got != "hello from fallback" {
		t.Fatalf("response text = %q, want translated Claude text; payload=%s", got, resp.Payload)
	}
	if gjson.GetBytes(resp.Payload, "response").Exists() {
		t.Fatalf("response retained raw Codex envelope: %s", resp.Payload)
	}
}

func TestCodexWebsocketsExecute426FallbackPreservesClaudePromptCacheSession(t *testing.T) {
	type capturedRequest struct {
		body      []byte
		sessionID string
	}
	captured := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			w.WriteHeader(http.StatusUpgradeRequired)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read fallback body: %v", err)
			return
		}
		captured <- capturedRequest{body: body, sessionID: r.Header.Get("Session_id")}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-5-codex\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}\n\n"))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model: "gpt-5-codex",
		Payload: []byte(`{
			"model":"gpt-5-codex",
			"max_tokens":128,
			"metadata":{"user_id":"{\"device_id\":\"device-a\",\"account_uuid\":\"\",\"session_id\":\"fallback-cache-session\"}"},
			"messages":[{"role":"user","content":"hello"}]
		}`),
	}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude}

	if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	select {
	case got := <-captured:
		promptCacheKey := gjson.GetBytes(got.body, "prompt_cache_key").String()
		if promptCacheKey == "" {
			t.Fatalf("fallback body missing prompt_cache_key: %s", got.body)
		}
		if got.sessionID != promptCacheKey {
			t.Fatalf("Session_id = %q, want prompt_cache_key %q", got.sessionID, promptCacheKey)
		}
		if gjson.GetBytes(got.body, "messages").Exists() {
			t.Fatalf("prepared payload was retransformed into Claude shape: %s", got.body)
		}
		if input := gjson.GetBytes(got.body, "input"); !input.Exists() || !strings.Contains(input.Raw, "hello") {
			t.Fatalf("prepared Codex input missing user content: %s", got.body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for HTTP fallback request")
	}
}

func TestCodexWebsocketsExecuteStream426FallbackPreservesImplicitClaudeResponseFormat(t *testing.T) {
	captured := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			w.WriteHeader(http.StatusUpgradeRequired)
			return
		}
		if got := r.Header.Get(codexResponsesLiteHeader); got != "true" {
			t.Errorf("%s = %q, want true", codexResponsesLiteHeader, got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read fallback body: %v", err)
			return
		}
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-5-codex\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello from fallback\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-5-codex\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n"))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"gpt-5-codex","max_tokens":128,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","description":"lookup","input_schema":{"type":"object","properties":{}}}],"stream":true}`),
	}
	opts := cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude,
		Headers:      http.Header{codexResponsesLiteHeader: []string{"true"}},
	}

	result, err := exec.ExecuteStream(context.Background(), auth, req, opts)
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	var output bytes.Buffer
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error = %v", chunk.Err)
		}
		output.Write(chunk.Payload)
	}
	stream := output.String()
	if !strings.Contains(stream, "event: message_start") || !strings.Contains(stream, `"type":"message_start"`) {
		t.Fatalf("stream missing translated Claude message_start: %s", stream)
	}
	if !strings.Contains(stream, `"type":"content_block_delta"`) || !strings.Contains(stream, `"text":"hello from fallback"`) {
		t.Fatalf("stream missing translated Claude text delta: %s", stream)
	}
	if strings.Contains(stream, `"type":"response.output_text.delta"`) || strings.Contains(stream, `"type":"response.completed"`) {
		t.Fatalf("stream retained raw Codex events: %s", stream)
	}

	select {
	case body := <-captured:
		if !gjson.ValidBytes(body) {
			t.Fatalf("fallback body is invalid JSON: %s", body)
		}
		assertCodexLiteUpstreamPayload(t, body)
		if gjson.GetBytes(body, "messages").Exists() {
			t.Fatalf("fallback body retained Claude messages instead of Codex input: %s", body)
		}
		if input := gjson.GetBytes(body, "input"); !input.Exists() || !strings.Contains(input.Raw, "hello") {
			t.Fatalf("fallback body missing translated Codex input: %s", body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for HTTP fallback request")
	}
}

func TestCodexWebsocketsExecuteStream426FallbackReleasesSessionLock(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
			w.WriteHeader(http.StatusUpgradeRequired)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"output\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}\n\n"))
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":"hello"}`)}
	opts := cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAIResponse,
		Metadata:     map[string]any{cliproxyexecutor.ExecutionSessionMetadataKey: "fallback-lock-session"},
	}

	run := func() error {
		result, err := exec.ExecuteStream(context.Background(), auth, req, opts)
		if err != nil {
			return err
		}
		for chunk := range result.Chunks {
			if chunk.Err != nil {
				return chunk.Err
			}
		}
		return nil
	}
	if err := run(); err != nil {
		t.Fatalf("first ExecuteStream() error = %v", err)
	}
	secondDone := make(chan error, 1)
	go func() { secondDone <- run() }()
	select {
	case err := <-secondDone:
		if err != nil {
			t.Fatalf("second ExecuteStream() error = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("second ExecuteStream() deadlocked; session request lock was not released")
	}
}

func TestCodexWebsocketsExecuteStreamPropagatesUpstreamErrorForDownstreamWebsocket(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	errorPayload := []byte(`{"type":"error","status":429,"error":{"code":"websocket_connection_limit_reached","message":"too many websockets"}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()

		if _, _, errRead := conn.ReadMessage(); errRead != nil {
			t.Errorf("read upstream websocket message: %v", errRead)
			return
		}
		if errWrite := conn.WriteMessage(websocket.TextMessage, errorPayload); errWrite != nil {
			t.Errorf("write error websocket message: %v", errWrite)
			return
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`),
	}
	opts := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FromString("openai-response"),
		ResponseFormat: sdktranslator.FromString("openai-response"),
	}
	ctx := cliproxyexecutor.WithDownstreamWebsocket(context.Background())

	result, err := exec.ExecuteStream(ctx, auth, req, opts)
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}

	select {
	case chunk, ok := <-result.Chunks:
		if !ok {
			t.Fatal("stream closed before error chunk")
		}
		if len(bytes.TrimSpace(chunk.Payload)) != 0 {
			t.Fatalf("error chunk payload = %q, want empty", chunk.Payload)
		}
		if chunk.Err == nil {
			t.Fatal("error chunk Err = nil, want upstream error")
		}
		statusErr, ok := chunk.Err.(interface{ StatusCode() int })
		if !ok {
			t.Fatalf("error type %T does not expose StatusCode", chunk.Err)
		}
		if got := statusErr.StatusCode(); got != http.StatusTooManyRequests {
			t.Fatalf("status = %d, want %d", got, http.StatusTooManyRequests)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for error stream chunk")
	}
}

func TestSendTerminalWebsocketReadInvalidatesBeforeWaitingForCapacity(t *testing.T) {
	terminalErr := &websocket.CloseError{Code: websocket.CloseMessageTooBig}

	t.Run("available channel keeps fast path ordering", func(t *testing.T) {
		ch := make(chan codexWebsocketRead, 1)
		done := make(chan struct{})
		invalidateCalls := 0
		invalidated := sendTerminalWebsocketRead(ch, done, codexWebsocketRead{err: terminalErr}, func() {
			invalidateCalls++
		})
		if invalidated {
			t.Fatal("available channel should not invalidate before delivery")
		}
		if invalidateCalls != 0 {
			t.Fatalf("invalidate calls = %d, want 0", invalidateCalls)
		}
		event := <-ch
		if !errors.Is(event.err, terminalErr) {
			t.Fatalf("terminal error = %v, want %v", event.err, terminalErr)
		}
	})

	t.Run("full channel invalidates before waiting", func(t *testing.T) {
		ch := make(chan codexWebsocketRead, 1)
		ch <- codexWebsocketRead{payload: []byte("queued")}
		done := make(chan struct{})
		invalidateCalled := make(chan struct{})
		result := make(chan bool, 1)

		go func() {
			result <- sendTerminalWebsocketRead(ch, done, codexWebsocketRead{err: terminalErr}, func() {
				close(invalidateCalled)
			})
		}()

		select {
		case <-invalidateCalled:
		case <-time.After(time.Second):
			t.Fatal("invalidation did not happen before waiting for channel capacity")
		}
		select {
		case <-result:
			t.Fatal("terminal sender returned before capacity was released")
		default:
		}

		<-ch
		select {
		case event := <-ch:
			if !errors.Is(event.err, terminalErr) {
				t.Fatalf("terminal error = %v, want %v", event.err, terminalErr)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for terminal read")
		}
		select {
		case invalidated := <-result:
			if !invalidated {
				t.Fatal("full channel should report early invalidation")
			}
		case <-time.After(time.Second):
			t.Fatal("terminal sender did not finish")
		}
	})

	t.Run("full channel stops when invalidation cancels active read", func(t *testing.T) {
		ch := make(chan codexWebsocketRead, 1)
		ch <- codexWebsocketRead{payload: []byte("queued")}
		done := make(chan struct{})
		invalidated := sendTerminalWebsocketRead(ch, done, codexWebsocketRead{err: terminalErr}, func() {
			close(done)
		})
		if !invalidated {
			t.Fatal("full channel should report early invalidation")
		}
		if len(ch) != 1 {
			t.Fatalf("channel length = %d, want queued payload only", len(ch))
		}
	})
}

func TestMapCodexWebsocketWriteErrorStopsRetryForMessageTooBig(t *testing.T) {
	networkWriteErr := errors.New("write: broken pipe")
	tests := []struct {
		name       string
		closeCode  int
		writeErr   error
		wantStatus int
		wantRetry  bool
	}{
		{
			name:       "close sent after message too big is request scoped",
			closeCode:  websocket.CloseMessageTooBig,
			writeErr:   websocket.ErrCloseSent,
			wantStatus: http.StatusRequestEntityTooLarge,
			wantRetry:  false,
		},
		{
			name:       "network write error after message too big is request scoped",
			closeCode:  websocket.CloseMessageTooBig,
			writeErr:   networkWriteErr,
			wantStatus: http.StatusRequestEntityTooLarge,
			wantRetry:  false,
		},
		{
			name:      "other close keeps stale connection retry",
			closeCode: websocket.CloseNormalClosure,
			writeErr:  websocket.ErrCloseSent,
			wantRetry: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sess := &codexWebsocketSession{}
			conn := &websocket.Conn{}
			sess.resetUpstreamDisconnectError(conn)
			sess.setUpstreamDisconnectError(conn, &websocket.CloseError{Code: tt.closeCode})

			mappedErr := mapCodexWebsocketWriteError(sess, conn, tt.writeErr)
			if got := shouldRetryCodexWebsocketSend(mappedErr); got != tt.wantRetry {
				t.Fatalf("shouldRetryCodexWebsocketSend() = %v, want %v; err=%v", got, tt.wantRetry, mappedErr)
			}
			if tt.wantStatus == 0 {
				if !errors.Is(mappedErr, tt.writeErr) {
					t.Fatalf("mapped error = %v, want %v", mappedErr, tt.writeErr)
				}
				return
			}
			statusErr, ok := mappedErr.(interface{ StatusCode() int })
			if !ok || statusErr.StatusCode() != tt.wantStatus {
				t.Fatalf("mapped status = %v, want %d; err=%v", statusErr, tt.wantStatus, mappedErr)
			}
			requestErr, ok := mappedErr.(interface{ IsRequestScoped() bool })
			if !ok || !requestErr.IsRequestScoped() {
				t.Fatalf("mapped error should be request scoped, got %T", mappedErr)
			}
		})
	}
}

func TestMapCodexWebsocketWriteErrorDoesNotReusePriorConnectionClose(t *testing.T) {
	sess := &codexWebsocketSession{}
	priorConn := &websocket.Conn{}
	replacementConn := &websocket.Conn{}

	sess.resetUpstreamDisconnectError(priorConn)
	sess.setUpstreamDisconnectError(priorConn, &websocket.CloseError{Code: websocket.CloseMessageTooBig})
	priorErr := mapCodexWebsocketWriteError(sess, priorConn, websocket.ErrCloseSent)
	if shouldRetryCodexWebsocketSend(priorErr) {
		t.Fatalf("prior connection 1009 should not retry, got %v", priorErr)
	}

	sess.resetUpstreamDisconnectError(replacementConn)
	// A late close callback from the prior connection must not overwrite the
	// replacement connection's close state.
	sess.setUpstreamDisconnectError(priorConn, &websocket.CloseError{Code: websocket.CloseMessageTooBig})
	sess.setUpstreamDisconnectError(replacementConn, &websocket.CloseError{Code: websocket.CloseNormalClosure})
	replacementErr := mapCodexWebsocketWriteError(sess, replacementConn, websocket.ErrCloseSent)
	if !errors.Is(replacementErr, websocket.ErrCloseSent) {
		t.Fatalf("replacement connection error = %v, want %v", replacementErr, websocket.ErrCloseSent)
	}
	if !shouldRetryCodexWebsocketSend(replacementErr) {
		t.Fatalf("replacement connection should keep stale-connection retry, got %v", replacementErr)
	}
}

func TestCodexWebsocketsExecuteStreamMapsMessageTooBigClose(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()

		if _, _, errRead := conn.ReadMessage(); errRead != nil {
			t.Errorf("read upstream websocket message: %v", errRead)
			return
		}
		deadline := time.Now().Add(time.Second)
		closeMessage := websocket.FormatCloseMessage(websocket.CloseMessageTooBig, "message too big")
		if errWrite := conn.WriteControl(websocket.CloseMessage, closeMessage, deadline); errWrite != nil {
			t.Errorf("write close websocket message: %v", errWrite)
			return
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`),
	}
	opts := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FromString("openai-response"),
		ResponseFormat: sdktranslator.FromString("openai-response"),
	}

	result, err := exec.ExecuteStream(context.Background(), auth, req, opts)
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}

	select {
	case chunk, ok := <-result.Chunks:
		if !ok {
			t.Fatal("stream closed before error chunk")
		}
		if chunk.Err == nil {
			t.Fatal("error chunk Err = nil, want message-too-big error")
		}
		statusErr, ok := chunk.Err.(interface{ StatusCode() int })
		if !ok {
			t.Fatalf("error type %T does not expose StatusCode", chunk.Err)
		}
		if got := statusErr.StatusCode(); got != http.StatusRequestEntityTooLarge {
			t.Fatalf("status = %d, want %d", got, http.StatusRequestEntityTooLarge)
		}
		if got := gjson.Get(chunk.Err.Error(), "error.code").String(); got != "message_too_big" {
			t.Fatalf("error code = %q, want message_too_big; err=%v", got, chunk.Err)
		}
		requestErr, ok := chunk.Err.(interface{ IsRequestScoped() bool })
		if !ok || !requestErr.IsRequestScoped() {
			t.Fatalf("message-too-big error should be request scoped, got %T", chunk.Err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for error stream chunk")
	}
}

func TestCodexWebsocketsUpstreamDisconnectChanSignalsOnInvalidate(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		for {
			if _, _, errRead := conn.ReadMessage(); errRead != nil {
				return
			}
		}
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial websocket: %v", err)
	}
	defer func() { _ = conn.Close() }()

	exec := NewCodexWebsocketsExecutor(&config.Config{})
	exec.store = &codexWebsocketSessionStore{sessions: make(map[string]*codexWebsocketSession)}
	sessionID := "sess-1"
	disconnectCh := exec.UpstreamDisconnectChan(sessionID)
	if disconnectCh == nil {
		t.Fatal("expected disconnect channel")
	}

	sess := exec.getOrCreateSession(sessionID)
	if sess == nil {
		t.Fatal("expected session")
	}
	sess.connMu.Lock()
	sess.conn = conn
	sess.authID = "auth-1"
	sess.wsURL = "ws://example.test/responses"
	sess.readerConn = conn
	sess.connMu.Unlock()

	upstreamErr := errors.New("upstream gone")
	exec.invalidateUpstreamConn(sess, conn, "test_invalidate", upstreamErr)

	select {
	case errRead, ok := <-disconnectCh:
		if !ok {
			t.Fatal("expected disconnect channel to deliver error before closing")
		}
		if errRead == nil || errRead.Error() != upstreamErr.Error() {
			t.Fatalf("disconnect error = %v, want %v", errRead, upstreamErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for disconnect signal")
	}
}

func TestApplyCodexWebsocketHeadersDefaultsToCurrentResponsesBeta(t *testing.T) {
	headers := applyCodexWebsocketHeaders(context.Background(), http.Header{}, nil, nil, "", nil)

	if got := headers.Get("OpenAI-Beta"); got != codexResponsesWebsocketBetaHeaderValue {
		t.Fatalf("OpenAI-Beta = %s, want %s", got, codexResponsesWebsocketBetaHeaderValue)
	}
	if got := headers.Get("User-Agent"); got != codexUserAgent {
		t.Fatalf("User-Agent = %s, want %s", got, codexUserAgent)
	}
	if !strings.HasPrefix(codexUserAgent, codexOriginator+"/") {
		t.Fatalf("default Codex User-Agent = %s, want prefix %s/", codexUserAgent, codexOriginator)
	}
	if !strings.HasPrefix(codexUserAgent, "codex-tui/") {
		t.Fatalf("default Codex User-Agent = %s, want codex-tui prefix", codexUserAgent)
	}
	if !strings.Contains(codexUserAgent, "(codex-tui;") {
		t.Fatalf("default Codex User-Agent = %s, want codex-tui suffix", codexUserAgent)
	}
	if got := headers.Get("Originator"); got != codexOriginator {
		t.Fatalf("Originator = %s, want %s", got, codexOriginator)
	}
	if got := headers.Get("Version"); got != "" {
		t.Fatalf("Version = %q, want empty", got)
	}
	if got := headers.Get("x-codex-beta-features"); got != "" {
		t.Fatalf("x-codex-beta-features = %q, want empty", got)
	}
	if got := headers.Get("X-Codex-Turn-Metadata"); got != "" {
		t.Fatalf("X-Codex-Turn-Metadata = %q, want empty", got)
	}
	if got := headers.Get("X-Client-Request-Id"); got != "" {
		t.Fatalf("X-Client-Request-Id = %q, want empty", got)
	}
}

func TestApplyCodexWebsocketHeadersPassesThroughClientIdentityHeaders(t *testing.T) {
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	ctx := contextWithGinHeaders(map[string]string{
		"Originator":            "Codex Desktop",
		"User-Agent":            "codex_cli_rs/0.1.0",
		"Version":               "0.115.0-alpha.27",
		"X-Codex-Turn-Metadata": `{"turn_id":"turn-1"}`,
		"X-Client-Request-Id":   "019d2233-e240-7162-992d-38df0a2a0e0d",
		"session-id":            "legacy-session",
	})

	headers := applyCodexWebsocketHeaders(ctx, http.Header{}, nil, auth, "", nil)

	if got := headers.Get("Originator"); got != "Codex Desktop" {
		t.Fatalf("Originator = %s, want %s", got, "Codex Desktop")
	}
	if got := headers.Get("User-Agent"); got != "codex_cli_rs/0.1.0" {
		t.Fatalf("User-Agent = %s, want %s", got, "codex_cli_rs/0.1.0")
	}
	if got := headers.Get("Version"); got != "0.115.0-alpha.27" {
		t.Fatalf("Version = %s, want %s", got, "0.115.0-alpha.27")
	}
	if got := headers.Get("X-Codex-Turn-Metadata"); got != `{"turn_id":"turn-1"}` {
		t.Fatalf("X-Codex-Turn-Metadata = %s, want %s", got, `{"turn_id":"turn-1"}`)
	}
	if got := headers.Get("X-Client-Request-Id"); got != "019d2233-e240-7162-992d-38df0a2a0e0d" {
		t.Fatalf("X-Client-Request-Id = %s, want %s", got, "019d2233-e240-7162-992d-38df0a2a0e0d")
	}
	if got := headers["session_id"]; len(got) != 1 || got[0] != "legacy-session" {
		t.Fatalf("session_id = %#v, want [legacy-session]", got)
	}
	if got := headers.Get("Session-Id"); got != "" {
		t.Fatalf("Session-Id = %s, want empty", got)
	}
}

func TestCodexWebsocketHeadersUseExecutionResponsesLiteHeaderAtFinalPrecedence(t *testing.T) {
	ctx := contextWithGinHeaders(map[string]string{
		codexResponsesLiteHeader: "false",
	})
	executionHeaders := http.Header{codexResponsesLiteHeader: []string{"TRUE"}}

	headers := applyCodexWebsocketHeaders(ctx, http.Header{}, executionHeaders, nil, "", nil)
	forwardCodexResponsesLiteHeader(headers, resolveCodexResponsesLite(nil, executionHeaders, ""))

	if got := headers.Get(codexResponsesLiteHeader); got != "TRUE" {
		t.Fatalf("%s = %q, want TRUE", codexResponsesLiteHeader, got)
	}
}

func TestCodexWebsocketHeadersDoNotInventResponsesLiteHeaderAtFinalPrecedence(t *testing.T) {
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"header:" + codexResponsesLiteHeader: "auth-custom",
	}}
	headers := applyCodexWebsocketHeaders(context.Background(), http.Header{}, nil, auth, "", nil)
	forwardCodexResponsesLiteHeader(headers, codexResponsesLiteDecision{})
	if got := headers.Get(codexResponsesLiteHeader); got != "" {
		t.Fatalf("%s = %q, want absent without execution header", codexResponsesLiteHeader, got)
	}
}

func TestApplyCodexWebsocketHeadersCanonicalizesLegacyUnderscoreSessionHeader(t *testing.T) {
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	ctx := contextWithGinHeaders(map[string]string{
		"Originator": "Codex Desktop",
		"User-Agent": "codex_cli_rs/0.1.0",
		"Session_id": "legacy-underscore-session",
	})

	headers := applyCodexWebsocketHeaders(ctx, http.Header{}, nil, auth, "", nil)

	if got := headers["session_id"]; len(got) != 1 || got[0] != "legacy-underscore-session" {
		t.Fatalf("session_id = %#v, want [legacy-underscore-session]", got)
	}
	if got := headers.Get("Session-Id"); got != "" {
		t.Fatalf("Session-Id = %s, want empty", got)
	}
}

func TestApplyCodexWebsocketHeadersUsesConfigDefaultsForOAuth(t *testing.T) {
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent:    "my-codex-client/1.0",
			BetaFeatures: "feature-a,feature-b",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}

	headers := applyCodexWebsocketHeaders(context.Background(), http.Header{}, nil, auth, "", cfg)

	if got := headers.Get("User-Agent"); got != "my-codex-client/1.0" {
		t.Fatalf("User-Agent = %s, want %s", got, "my-codex-client/1.0")
	}
	if got := headers.Get("x-codex-beta-features"); got != "feature-a,feature-b" {
		t.Fatalf("x-codex-beta-features = %s, want %s", got, "feature-a,feature-b")
	}
	if got := headers.Get("OpenAI-Beta"); got != codexResponsesWebsocketBetaHeaderValue {
		t.Fatalf("OpenAI-Beta = %s, want %s", got, codexResponsesWebsocketBetaHeaderValue)
	}
}

func TestApplyCodexWebsocketHeadersPrefersExistingHeadersOverClientAndConfig(t *testing.T) {
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent:    "config-ua",
			BetaFeatures: "config-beta",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	ctx := contextWithGinHeaders(map[string]string{
		"User-Agent":            "client-ua",
		"X-Codex-Beta-Features": "client-beta",
	})
	headers := http.Header{}
	headers.Set("User-Agent", "existing-ua")
	headers.Set("X-Codex-Beta-Features", "existing-beta")

	got := applyCodexWebsocketHeaders(ctx, headers, nil, auth, "", cfg)

	if gotVal := got.Get("User-Agent"); gotVal != "existing-ua" {
		t.Fatalf("User-Agent = %s, want %s", gotVal, "existing-ua")
	}
	if gotVal := got.Get("x-codex-beta-features"); gotVal != "existing-beta" {
		t.Fatalf("x-codex-beta-features = %s, want %s", gotVal, "existing-beta")
	}
}

func TestApplyCodexWebsocketHeadersConfigUserAgentOverridesClientHeader(t *testing.T) {
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent:    "config-ua",
			BetaFeatures: "config-beta",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	ctx := contextWithGinHeaders(map[string]string{
		"User-Agent":            "client-ua",
		"X-Codex-Beta-Features": "client-beta",
	})

	headers := applyCodexWebsocketHeaders(ctx, http.Header{}, nil, auth, "", cfg)

	if got := headers.Get("User-Agent"); got != "config-ua" {
		t.Fatalf("User-Agent = %s, want %s", got, "config-ua")
	}
	if got := headers.Get("x-codex-beta-features"); got != "client-beta" {
		t.Fatalf("x-codex-beta-features = %s, want %s", got, "client-beta")
	}
}

func TestApplyCodexWebsocketHeadersIgnoresConfigForAPIKeyAuth(t *testing.T) {
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent:    "config-ua",
			BetaFeatures: "config-beta",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider:   "codex",
		Attributes: map[string]string{"api_key": "sk-test"},
	}

	headers := applyCodexWebsocketHeaders(context.Background(), http.Header{}, nil, auth, "sk-test", cfg)

	if got := headers.Get("User-Agent"); got != "" {
		t.Fatalf("User-Agent = %s, want empty", got)
	}
	if got := headers.Get("x-codex-beta-features"); got != "" {
		t.Fatalf("x-codex-beta-features = %q, want empty", got)
	}
	if got := headers.Get("Originator"); got != "" {
		t.Fatalf("Originator = %s, want empty", got)
	}
}

func TestApplyCodexWebsocketHeadersPreservesExplicitAPIKeyUserAgent(t *testing.T) {
	auth := &cliproxyauth.Auth{Provider: "codex", Attributes: map[string]string{"api_key": "sk-test"}}
	ctx := contextWithGinHeaders(map[string]string{"User-Agent": "api-key-client/1.0", "Originator": "explicit-origin"})

	headers := applyCodexWebsocketHeaders(ctx, http.Header{}, nil, auth, "sk-test", nil)

	if got := headers.Get("User-Agent"); got != "api-key-client/1.0" {
		t.Fatalf("User-Agent = %s, want api-key-client/1.0", got)
	}
	if got := headers.Get("Originator"); got != "explicit-origin" {
		t.Fatalf("Originator = %s, want explicit-origin", got)
	}
}

func TestApplyCodexWebsocketHeadersUsesCanonicalAccountHeader(t *testing.T) {
	auth := &cliproxyauth.Auth{Provider: "codex", Metadata: map[string]any{"account_id": "acct-1"}}

	headers := applyCodexWebsocketHeaders(context.Background(), http.Header{}, nil, auth, "", nil)

	if got := headerValueCaseInsensitive(headers, "ChatGPT-Account-ID"); got != "acct-1" {
		t.Fatalf("ChatGPT-Account-ID = %s, want acct-1", got)
	}
	values, ok := headers["ChatGPT-Account-ID"]
	if !ok {
		t.Fatalf("expected exact ChatGPT-Account-ID key, got %#v", headers)
	}
	if len(values) != 1 || values[0] != "acct-1" {
		t.Fatalf("ChatGPT-Account-ID values = %#v, want [acct-1]", values)
	}
}

func TestApplyCodexPromptCacheHeadersSetsSessionIDAndLegacyConversation(t *testing.T) {
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"prompt_cache_key":"cache-1"}`)}

	_, headers := applyCodexPromptCacheHeaders("openai-response", req, []byte(`{"model":"gpt-5-codex"}`))

	if got := headers["session_id"]; len(got) != 1 || got[0] != "cache-1" {
		t.Fatalf("session_id = %#v, want [cache-1]", got)
	}
	if got := headers.Get("Session-Id"); got != "" {
		t.Fatalf("Session-Id = %s, want empty", got)
	}
	if got := headers.Get("Conversation_id"); got != "cache-1" {
		t.Fatalf("Conversation_id = %s, want cache-1", got)
	}
}

func TestApplyCodexPromptCacheHeadersUsesDerivedSessionUUID(t *testing.T) {
	t.Parallel()

	req := cliproxyexecutor.Request{
		Model:    "gpt-5-codex",
		Payload:  []byte(`{"input":"hello"}`),
		Metadata: map[string]any{cliproxyexecutor.DerivedSessionIDMetadataKey: "ctx:v1:derived-root"},
	}
	body, headers := applyCodexPromptCacheHeaders(sdktranslator.FormatInteractions, req, []byte(`{"model":"gpt-5-codex"}`))
	cacheKey := gjson.GetBytes(body, "prompt_cache_key").String()
	if _, errParse := uuid.Parse(cacheKey); errParse != nil {
		t.Fatalf("prompt_cache_key %q is not a UUID: %v", cacheKey, errParse)
	}
	if got := headers["session_id"]; len(got) != 1 || got[0] != cacheKey {
		t.Fatalf("session_id = %#v, want [%q]", got, cacheKey)
	}
	if got := headers.Get("Conversation_id"); got != cacheKey {
		t.Fatalf("Conversation_id = %q, want %q", got, cacheKey)
	}
}

func TestApplyCodexPromptCacheHeadersKeepsExecutionSessionAcrossIncrementalRoots(t *testing.T) {
	t.Parallel()

	firstReq := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"input":"first"}`),
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: "connection-1",
			cliproxyexecutor.DerivedSessionIDMetadataKey: "ctx:v1:first-root",
		},
	}
	secondReq := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"input":"second"}`),
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: "connection-1",
			cliproxyexecutor.DerivedSessionIDMetadataKey: "ctx:v1:second-root",
		},
	}
	firstBody, _ := applyCodexPromptCacheHeaders(sdktranslator.FormatOpenAIResponse, firstReq, []byte(`{"model":"gpt-5-codex"}`))
	secondBody, _ := applyCodexPromptCacheHeaders(sdktranslator.FormatOpenAIResponse, secondReq, []byte(`{"model":"gpt-5-codex"}`))
	firstKey := gjson.GetBytes(firstBody, "prompt_cache_key").String()
	secondKey := gjson.GetBytes(secondBody, "prompt_cache_key").String()
	if firstKey == "" || firstKey != secondKey {
		t.Fatalf("incremental websocket roots changed prompt cache key: first=%q second=%q", firstKey, secondKey)
	}
}

func TestApplyCodexPromptCacheHeadersClaudeUsesClaudeCodeSessionID(t *testing.T) {
	firstReq := cliproxyexecutor.Request{
		Model: "gpt-5-codex-claude-ws-cache-session",
		Payload: []byte(`{
			"metadata":{"user_id":"{\"device_id\":\"device-a\",\"account_uuid\":\"\",\"session_id\":\"ws-cache-session-1\"}"},
			"messages":[{"role":"user","content":[{"type":"text","text":"first"}]}]
		}`),
	}
	secondReq := cliproxyexecutor.Request{
		Model: "gpt-5-codex-claude-ws-cache-session",
		Payload: []byte(`{
			"metadata":{"user_id":"{\"device_id\":\"device-b\",\"account_uuid\":\"\",\"session_id\":\"ws-cache-session-1\"}"},
			"messages":[{"role":"user","content":[{"type":"text","text":"next"}]}]
		}`),
	}

	firstBody, firstHeaders := applyCodexPromptCacheHeaders("claude", firstReq, []byte(`{"model":"gpt-5-codex"}`))
	secondBody, secondHeaders := applyCodexPromptCacheHeaders("claude", secondReq, []byte(`{"model":"gpt-5-codex"}`))

	firstKey := gjson.GetBytes(firstBody, "prompt_cache_key").String()
	secondKey := gjson.GetBytes(secondBody, "prompt_cache_key").String()
	if firstKey == "" {
		t.Fatalf("first prompt_cache_key is empty; body=%s", string(firstBody))
	}
	if secondKey != firstKey {
		t.Fatalf("same Claude Code session_id produced different websocket prompt_cache_key: first=%q second=%q", firstKey, secondKey)
	}
	if got := firstHeaders["session_id"]; len(got) != 1 || got[0] != firstKey {
		t.Fatalf("first session_id = %#v, want [%q]", got, firstKey)
	}
	if got := secondHeaders["session_id"]; len(got) != 1 || got[0] != firstKey {
		t.Fatalf("second session_id = %#v, want [%q]", got, firstKey)
	}
}

func TestApplyCodexPromptCacheHeadersClaudeRejectsBareUserID(t *testing.T) {
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex-claude-ws-cache-bare-user",
		Payload: []byte(`{"metadata":{"user_id":"same-user-across-chats"},"messages":[{"role":"user","content":[{"type":"text","text":"first"}]}]}`),
	}

	body, headers := applyCodexPromptCacheHeaders("claude", req, []byte(`{"model":"gpt-5-codex"}`))

	if got := gjson.GetBytes(body, "prompt_cache_key").String(); got != "" {
		t.Fatalf("bare metadata.user_id must not create websocket prompt_cache_key, got %q; body=%s", got, string(body))
	}
	if got := headers["session_id"]; len(got) != 0 {
		t.Fatalf("bare metadata.user_id must not create websocket session_id, got %#v", got)
	}
	if got := headers.Get("Session-Id"); got != "" {
		t.Fatalf("bare metadata.user_id must not create websocket Session-Id, got %q", got)
	}
	if got := headers.Get("Conversation_id"); got != "" {
		t.Fatalf("bare metadata.user_id must not create websocket Conversation_id, got %q", got)
	}
}

func TestApplyCodexWebsocketHeadersIdentityConfuseRemapsPromptCacheKey(t *testing.T) {
	cfg := &config.Config{
		Routing: config.RoutingConfig{SessionAffinity: true},
		Codex:   config.CodexConfig{IdentityConfuse: true},
	}
	auth := &cliproxyauth.Auth{ID: "auth-ws-1", Provider: "codex"}
	req := cliproxyexecutor.Request{
		Model:   "gpt-5-codex",
		Payload: []byte(`{"prompt_cache_key":"cache-ws-1","client_metadata":{"x-codex-installation-id":"install-ws-1"}}`),
	}

	body, headers := applyCodexPromptCacheHeaders("openai-response", req, []byte(`{"model":"gpt-5-codex"}`))
	body, identityState := applyCodexIdentityConfuseBody(cfg, auth, req.Payload, body)
	ctx := contextWithGinHeaders(map[string]string{
		"X-Codex-Turn-Metadata": `{"prompt_cache_key":"cache-ws-1","turn_id":"turn-ws-1","window_id":"cache-ws-1:0"}`,
		"X-Client-Request-Id":   "client-request-1",
	})
	headers = applyCodexWebsocketHeaders(ctx, headers, nil, auth, "oauth-token", cfg)
	applyCodexIdentityConfuseHeaders(headers, &identityState)

	expectedPromptCacheKey := codexIdentityConfuseUUID("auth-ws-1", "prompt-cache", "cache-ws-1")
	expectedTurnID := codexIdentityConfuseUUID("auth-ws-1", "turn", "turn-ws-1")
	if gotKey := gjson.GetBytes(body, "prompt_cache_key").String(); gotKey != expectedPromptCacheKey {
		t.Fatalf("prompt_cache_key = %q, want %q", gotKey, expectedPromptCacheKey)
	}
	if gotSession := headers["session_id"]; len(gotSession) != 1 || gotSession[0] != expectedPromptCacheKey {
		t.Fatalf("session_id = %#v, want [%q]", gotSession, expectedPromptCacheKey)
	}
	if gotCanonicalSession := headers.Get("Session-Id"); gotCanonicalSession != "" {
		t.Fatalf("Session-Id = %q, want empty", gotCanonicalSession)
	}
	if gotRequestID := headers.Get("X-Client-Request-Id"); gotRequestID != expectedPromptCacheKey {
		t.Fatalf("X-Client-Request-Id = %q, want %q", gotRequestID, expectedPromptCacheKey)
	}
	if gotThreadID := headers.Get("Thread-Id"); gotThreadID != expectedPromptCacheKey {
		t.Fatalf("Thread-Id = %q, want %q", gotThreadID, expectedPromptCacheKey)
	}
	if gotConversation := headers.Get("Conversation_id"); gotConversation != expectedPromptCacheKey {
		t.Fatalf("Conversation_id = %q, want %q", gotConversation, expectedPromptCacheKey)
	}
	if gotWindowID := headers.Get("X-Codex-Window-Id"); gotWindowID != expectedPromptCacheKey+":0" {
		t.Fatalf("X-Codex-Window-Id = %q, want %q", gotWindowID, expectedPromptCacheKey+":0")
	}
	gotMetadata := headers.Get("X-Codex-Turn-Metadata")
	if gotMetadataPromptCacheKey := gjson.Get(gotMetadata, "prompt_cache_key").String(); gotMetadataPromptCacheKey != expectedPromptCacheKey {
		t.Fatalf("X-Codex-Turn-Metadata.prompt_cache_key = %q, want %q", gotMetadataPromptCacheKey, expectedPromptCacheKey)
	}
	if gotMetadataTurnID := gjson.Get(gotMetadata, "turn_id").String(); gotMetadataTurnID != expectedTurnID {
		t.Fatalf("X-Codex-Turn-Metadata.turn_id = %q, want %q", gotMetadataTurnID, expectedTurnID)
	}
	if gotMetadataWindowID := gjson.Get(gotMetadata, "window_id").String(); gotMetadataWindowID != expectedPromptCacheKey+":0" {
		t.Fatalf("X-Codex-Turn-Metadata.window_id = %q, want %q", gotMetadataWindowID, expectedPromptCacheKey+":0")
	}
	expectedInstallationID := codexIdentityConfuseUUID("auth-ws-1", "installation", "install-ws-1")
	if gotInstallationID := gjson.GetBytes(body, "client_metadata.x-codex-installation-id").String(); gotInstallationID != expectedInstallationID {
		t.Fatalf("installation id = %q, want %q", gotInstallationID, expectedInstallationID)
	}
}

func TestCodexIdentityConfuseResponsePayloadHidesUpstreamAndRestoresClient(t *testing.T) {
	state := codexIdentityConfuseState{
		enabled:                true,
		authID:                 "auth-ws-1",
		originalPromptCacheKey: "cache-ws-1",
		promptCacheKey:         codexIdentityConfuseUUID("auth-ws-1", "prompt-cache", "cache-ws-1"),
	}
	expectedTurnID := state.confuseTurnID("turn-ws-1")
	rawPayload := []byte(`{"type":"response.completed","response":{"prompt_cache_key":"cache-ws-1","turn_id":"turn-ws-1"},"prompt_cache_key":"cache-ws-1","turn_id":"turn-ws-1"}`)

	upstreamPayload := applyCodexIdentityConfuseResponsePayload(rawPayload, state)
	if bytes.Contains(upstreamPayload, []byte(`cache-ws-1`)) {
		t.Fatalf("upstream payload still contains original prompt_cache_key: %s", string(upstreamPayload))
	}
	if bytes.Contains(upstreamPayload, []byte(`turn-ws-1`)) {
		t.Fatalf("upstream payload still contains original turn_id: %s", string(upstreamPayload))
	}
	if !bytes.Contains(upstreamPayload, []byte(state.promptCacheKey)) {
		t.Fatalf("upstream payload missing confused prompt_cache_key: %s", string(upstreamPayload))
	}
	if !bytes.Contains(upstreamPayload, []byte(expectedTurnID)) {
		t.Fatalf("upstream payload missing confused turn_id: %s", string(upstreamPayload))
	}

	clientPayload := applyCodexIdentityExposeResponsePayload(upstreamPayload, state)
	if bytes.Contains(clientPayload, []byte(state.promptCacheKey)) {
		t.Fatalf("client payload still contains confused prompt_cache_key: %s", string(clientPayload))
	}
	if bytes.Contains(clientPayload, []byte(expectedTurnID)) {
		t.Fatalf("client payload still contains confused turn_id: %s", string(clientPayload))
	}
	if !bytes.Contains(clientPayload, []byte(`cache-ws-1`)) {
		t.Fatalf("client payload missing original prompt_cache_key: %s", string(clientPayload))
	}
	if !bytes.Contains(clientPayload, []byte(`turn-ws-1`)) {
		t.Fatalf("client payload missing original turn_id: %s", string(clientPayload))
	}

	rawSSE := []byte(`data: {"type":"response.completed","response":{"prompt_cache_key":"cache-ws-1","turn_id":"turn-ws-1"}}`)
	upstreamSSE := applyCodexIdentityConfuseResponsePayload(rawSSE, state)
	if bytes.Contains(upstreamSSE, []byte(`cache-ws-1`)) {
		t.Fatalf("upstream SSE still contains original prompt_cache_key: %s", string(upstreamSSE))
	}
	if bytes.Contains(upstreamSSE, []byte(`turn-ws-1`)) {
		t.Fatalf("upstream SSE still contains original turn_id: %s", string(upstreamSSE))
	}
	clientSSE := applyCodexIdentityExposeResponsePayload(upstreamSSE, state)
	if !bytes.Contains(clientSSE, []byte(`cache-ws-1`)) || bytes.Contains(clientSSE, []byte(state.promptCacheKey)) {
		t.Fatalf("client SSE prompt_cache_key was not restored: %s", string(clientSSE))
	}
	if !bytes.Contains(clientSSE, []byte(`turn-ws-1`)) || bytes.Contains(clientSSE, []byte(expectedTurnID)) {
		t.Fatalf("client SSE turn_id was not restored: %s", string(clientSSE))
	}
}

func TestBuildCodexResponsesWebsocketURLRequiresHTTPURL(t *testing.T) {
	if got, err := buildCodexResponsesWebsocketURL("https://example.com/backend/responses"); err != nil || got != "wss://example.com/backend/responses" {
		t.Fatalf("https URL = %q, %v; want wss URL", got, err)
	}
	if _, err := buildCodexResponsesWebsocketURL("ftp://example.com/responses"); err == nil {
		t.Fatalf("expected unsupported scheme error")
	}
	if _, err := buildCodexResponsesWebsocketURL("https:///responses"); err == nil {
		t.Fatalf("expected empty host error")
	}
}

func TestParseCodexWebsocketErrorMarksConnectionLimitRetryable(t *testing.T) {
	err, ok := parseCodexWebsocketError([]byte(`{"type":"error","status":429,"error":{"code":"websocket_connection_limit_reached","message":"too many websockets"},"headers":{"retry-after":"1"}}`))
	if !ok {
		t.Fatalf("expected websocket error")
	}
	status, ok := err.(interface{ StatusCode() int })
	if !ok || status.StatusCode() != http.StatusTooManyRequests {
		t.Fatalf("status = %#v, want 429", err)
	}
	retryable, ok := err.(interface{ RetryAfter() *time.Duration })
	if !ok || retryable.RetryAfter() == nil {
		t.Fatalf("expected retryable websocket connection limit error")
	}
	if got := *retryable.RetryAfter(); got != 0 {
		t.Fatalf("retryAfter = %v, want connection-limit fallback 0", got)
	}
	withHeaders, ok := err.(interface{ Headers() http.Header })
	if !ok || withHeaders.Headers().Get("retry-after") != "1" {
		t.Fatalf("headers = %#v, want retry-after", err)
	}
}

func TestParseCodexWebsocketErrorUsesUsageLimitRetryMetadata(t *testing.T) {
	err, ok := parseCodexWebsocketError([]byte(`{"type":"error","status":429,"body":{"error":{"type":"usage_limit_reached","message":"usage limit reached","resets_in_seconds":7}}}`))
	if !ok {
		t.Fatalf("expected websocket error")
	}

	retryable, ok := err.(interface{ RetryAfter() *time.Duration })
	if !ok || retryable.RetryAfter() == nil {
		t.Fatalf("expected retryable usage limit websocket error")
	}
	if got := *retryable.RetryAfter(); got != 7*time.Second {
		t.Fatalf("retryAfter = %v, want 7s", got)
	}
}

func TestParseCodexWebsocketErrorPreservesWrappedBodyAndHeaders(t *testing.T) {
	err, ok := parseCodexWebsocketError([]byte(`{"type":"error","status":429,"body":{"error":{"code":"websocket_connection_limit_reached","type":"server_error","message":"too many websocket connections"}},"headers":{"x-request-id":"req-1"}}`))
	if !ok {
		t.Fatalf("expected websocket error")
	}

	parsed := gjson.Parse(err.Error())
	if got := parsed.Get("status").Int(); got != http.StatusTooManyRequests {
		t.Fatalf("wrapped status = %d, want 429; payload=%s", got, err.Error())
	}
	if got := parsed.Get("body.error.code").String(); got != "websocket_connection_limit_reached" {
		t.Fatalf("wrapped body error code = %s, want websocket_connection_limit_reached; payload=%s", got, err.Error())
	}
	if got := parsed.Get("error.code").String(); got != "websocket_connection_limit_reached" {
		t.Fatalf("surface error code = %s, want websocket_connection_limit_reached; payload=%s", got, err.Error())
	}
	retryable, ok := err.(interface{ RetryAfter() *time.Duration })
	if !ok || retryable.RetryAfter() == nil {
		t.Fatalf("expected body.error.code websocket connection limit to be retryable")
	}
	withHeaders, ok := err.(interface{ Headers() http.Header })
	if !ok || withHeaders.Headers().Get("x-request-id") != "req-1" {
		t.Fatalf("headers = %#v, want x-request-id", err)
	}
}

func TestApplyCodexHeadersUsesConfigUserAgentForOAuth(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://example.com/responses", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent:    "config-ua",
			BetaFeatures: "config-beta",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	req = req.WithContext(contextWithGinHeaders(map[string]string{
		"User-Agent": "client-ua",
	}))

	applyCodexHeaders(req, auth, "oauth-token", true, cfg)

	if got := req.Header.Get("User-Agent"); got != "config-ua" {
		t.Fatalf("User-Agent = %s, want %s", got, "config-ua")
	}
	if got := req.Header.Get("x-codex-beta-features"); got != "" {
		t.Fatalf("x-codex-beta-features = %q, want empty", got)
	}
}

func TestApplyModelHeaderOverridesFromModelConfig(t *testing.T) {
	const wantUA = "codex-tui/0.144.0 (Mac OS 26.5.1; arm64) iTerm.app/3.6.11 (codex-tui; 0.144.0)"
	req, err := http.NewRequest(http.MethodPost, "https://example.com/responses", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	cfg := &config.Config{
		CodexHeaderDefaults: config.CodexHeaderDefaults{
			UserAgent: "config-ua",
		},
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}

	applyCodexHeaders(req, auth, "oauth-token", true, cfg)
	applyModelHeaderOverrides(req.Header, "gpt-5.6-luna")

	if got := req.Header.Get("User-Agent"); got != wantUA {
		t.Fatalf("User-Agent = %q, want %q", got, wantUA)
	}
	if got := codexSessionHeaderValue(req.Header); got == "" {
		t.Fatal("expected Session_id to be set for Mac OS User-Agent override")
	}

	applyModelHeaderOverrides(req.Header, "gpt-5.4")
	if got := req.Header.Get("User-Agent"); got != wantUA {
		t.Fatalf("User-Agent after no-op override = %q, want %q", got, wantUA)
	}
}

func TestApplyModelHeaderOverridesMultipleHeaders(t *testing.T) {
	reg := registry.GetGlobalRegistry()
	clientID := "test-model-header-override"
	reg.RegisterClient(clientID, "codex", []*registry.ModelInfo{{
		ID: "test-override-headers-model",
		Config: &registry.ModelConfig{
			OverrideHeader: map[string]string{
				"user-agent":    "custom-ua/1.0",
				"originator":    "custom-origin",
				"x-test-header": "forced-value",
			},
		},
	}})
	t.Cleanup(func() { reg.UnregisterClient(clientID) })

	headers := http.Header{}
	headers.Set("User-Agent", "old-ua")
	headers.Set("Originator", "old-origin")
	headers.Set("X-Test-Header", "old-value")

	applyModelHeaderOverrides(headers, "test-override-headers-model")

	if got := headers.Get("User-Agent"); got != "custom-ua/1.0" {
		t.Fatalf("User-Agent = %q, want custom-ua/1.0", got)
	}
	if got := headers.Get("Originator"); got != "custom-origin" {
		t.Fatalf("Originator = %q, want custom-origin", got)
	}
	if got := headers.Get("X-Test-Header"); got != "forced-value" {
		t.Fatalf("X-Test-Header = %q, want forced-value", got)
	}
}

func TestApplyCodexHeadersPassesThroughClientIdentityHeaders(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://example.com/responses", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	auth := &cliproxyauth.Auth{
		Provider: "codex",
		Metadata: map[string]any{"email": "user@example.com"},
	}
	req = req.WithContext(contextWithGinHeaders(map[string]string{
		"Originator":            "Codex Desktop",
		"Version":               "0.115.0-alpha.27",
		"X-Codex-Turn-Metadata": `{"turn_id":"turn-1"}`,
		"X-Client-Request-Id":   "019d2233-e240-7162-992d-38df0a2a0e0d",
	}))

	applyCodexHeaders(req, auth, "oauth-token", true, nil)

	if got := req.Header.Get("Originator"); got != "Codex Desktop" {
		t.Fatalf("Originator = %s, want %s", got, "Codex Desktop")
	}
	if got := req.Header.Get("Version"); got != "0.115.0-alpha.27" {
		t.Fatalf("Version = %s, want %s", got, "0.115.0-alpha.27")
	}
	if got := req.Header.Get("X-Codex-Turn-Metadata"); got != `{"turn_id":"turn-1"}` {
		t.Fatalf("X-Codex-Turn-Metadata = %s, want %s", got, `{"turn_id":"turn-1"}`)
	}
	if got := req.Header.Get("X-Client-Request-Id"); got != "019d2233-e240-7162-992d-38df0a2a0e0d" {
		t.Fatalf("X-Client-Request-Id = %s, want %s", got, "019d2233-e240-7162-992d-38df0a2a0e0d")
	}
}

func TestApplyCodexHeadersDoesNotInjectClientOnlyHeadersByDefault(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://example.com/responses", nil)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}

	applyCodexHeaders(req, nil, "oauth-token", true, nil)

	if got := req.Header.Get("Version"); got != "" {
		t.Fatalf("Version = %q, want empty", got)
	}
	if got := req.Header.Get("X-Codex-Turn-Metadata"); got != "" {
		t.Fatalf("X-Codex-Turn-Metadata = %q, want empty", got)
	}
	if got := req.Header.Get("X-Client-Request-Id"); got != "" {
		t.Fatalf("X-Client-Request-Id = %q, want empty", got)
	}
}

func contextWithGinHeaders(headers map[string]string) context.Context {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	ginCtx.Request.Header = make(http.Header, len(headers))
	for key, value := range headers {
		ginCtx.Request.Header.Set(key, value)
	}
	return context.WithValue(context.Background(), "gin", ginCtx)
}

func TestNewProxyAwareWebsocketDialerDirectDisablesProxy(t *testing.T) {
	t.Parallel()

	dialer := newProxyAwareWebsocketDialer(
		&config.Config{SDKConfig: sdkconfig.SDKConfig{ProxyURL: "http://global-proxy.example.com:8080"}},
		&cliproxyauth.Auth{ProxyURL: "direct"},
	)

	if dialer.Proxy != nil {
		t.Fatal("expected websocket proxy function to be nil for direct mode")
	}
}

func TestCodexWebsocketUpgradeRequiredDoesNotFallbackToHTTPWithLifecycle(t *testing.T) {
	var httpFallbackCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			httpFallbackCalls.Add(1)
			http.Error(w, "unexpected HTTP fallback", http.StatusInternalServerError)
			return
		}
		http.Error(w, "websocket upgrade required", http.StatusUpgradeRequired)
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`)}
	opts := cliproxyexecutor.Options{
		SourceFormat:       sdktranslator.FromString("openai-response"),
		ResponseFormat:     sdktranslator.FromString("openai-response"),
		ExecutionLifecycle: newTerminalFailureLifecycle(),
	}

	if _, errExecute := exec.ExecuteStream(context.Background(), auth, req, opts); errExecute == nil {
		t.Fatal("ExecuteStream() error = nil, want failed Home lifecycle attempt")
	}
	if got := httpFallbackCalls.Load(); got != 0 {
		t.Fatalf("HTTP fallback calls = %d, want 0 with an execution lifecycle", got)
	}
}

func TestCodexWebsocketHandshakeFailureReleasesSessionRequestLock(t *testing.T) {
	for _, statusCode := range []int{http.StatusUpgradeRequired, http.StatusBadGateway} {
		t.Run(http.StatusText(statusCode), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "upstream rejected websocket", statusCode)
			}))
			defer server.Close()

			exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
			exec.store = &codexWebsocketSessionStore{sessions: make(map[string]*codexWebsocketSession)}
			auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
			req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`)}
			opts := cliproxyexecutor.Options{
				SourceFormat:   sdktranslator.FromString("openai-response"),
				ResponseFormat: sdktranslator.FromString("openai-response"),
				Metadata: map[string]any{
					cliproxyexecutor.ExecutionSessionMetadataKey: "failed-handshake",
				},
			}

			_, _ = exec.ExecuteStream(context.Background(), auth, req, opts)
			sess := exec.getOrCreateSession("failed-handshake")
			acquired := make(chan struct{})
			go func() {
				sess.reqMu.Lock()
				close(acquired)
				sess.reqMu.Unlock()
			}()
			select {
			case <-acquired:
			case <-time.After(time.Second):
				t.Fatal("websocket handshake failure left the session request lock held")
			}
		})
	}
}

type terminalFailureLifecycle struct {
	active atomic.Bool
	ends   atomic.Int32
}

func newTerminalFailureLifecycle() *terminalFailureLifecycle {
	lifecycle := &terminalFailureLifecycle{}
	lifecycle.active.Store(true)
	return lifecycle
}

func (*terminalFailureLifecycle) Bind(func() error) error { return nil }
func (l *terminalFailureLifecycle) End(string) {
	l.ends.Add(1)
	l.active.Store(false)
}
func (*terminalFailureLifecycle) Retain() {}

func TestCodexWebsocketTerminalFailureInvalidatesRetainedLifecycle(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	var connections atomic.Int32
	firstRelease := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Errorf("upgrade websocket: %v", errUpgrade)
			return
		}
		defer func() { _ = conn.Close() }()
		connection := connections.Add(1)
		if _, _, errRead := conn.ReadMessage(); errRead != nil {
			return
		}
		terminal := []byte(`{"type":"response.failed","response":{"error":{"type":"authentication_error","code":"invalid_api_key","message":"Invalid token."}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, terminal); errWrite != nil {
			t.Errorf("write terminal response: %v", errWrite)
		}
		if connection == 1 {
			<-firstRelease
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	exec.store = &codexWebsocketSessionStore{sessions: make(map[string]*codexWebsocketSession)}
	auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`)}
	opts := cliproxyexecutor.Options{
		SourceFormat:       sdktranslator.FromString("openai-response"),
		ResponseFormat:     sdktranslator.FromString("openai-response"),
		ExecutionLifecycle: newTerminalFailureLifecycle(),
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: "terminal-failure",
		},
	}

	result, errExecute := exec.ExecuteStream(context.Background(), auth, req, opts)
	if errExecute != nil {
		t.Fatalf("first ExecuteStream() error = %v", errExecute)
	}
	for chunk := range result.Chunks {
		if chunk.Err == nil {
			continue
		}
	}
	lifecycle := opts.ExecutionLifecycle.(*terminalFailureLifecycle)
	if lifecycle.active.Load() {
		t.Fatal("terminal failure left the retained lifecycle active")
	}
	if got := lifecycle.ends.Load(); got != 1 {
		t.Fatalf("retained lifecycle End calls = %d, want 1", got)
	}
	sess := exec.getOrCreateSession("terminal-failure")
	sess.connMu.Lock()
	connected := sess.conn != nil
	sess.connMu.Unlock()
	if connected {
		t.Fatal("terminal failure left the upstream session connection cached")
	}
	close(firstRelease)

	opts.ExecutionLifecycle = newTerminalFailureLifecycle()
	result, errExecute = exec.ExecuteStream(context.Background(), auth, req, opts)
	if errExecute != nil {
		t.Fatalf("second ExecuteStream() error = %v", errExecute)
	}
	for range result.Chunks {
	}
	if got := connections.Load(); got != 2 {
		t.Fatalf("websocket connections = %d, want 2 after terminal invalidation", got)
	}
}

type rejectingExecutionLifecycle struct{}

func (rejectingExecutionLifecycle) Bind(func() error) error {
	return errors.New("lifecycle bind rejected")
}
func (rejectingExecutionLifecycle) End(string) {}

func TestCodexWebsocketNonstreamLifecycleBindFailureDetachesConnection(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	var connections atomic.Int32
	closed := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Errorf("upgrade websocket: %v", errUpgrade)
			return
		}
		connection := connections.Add(1)
		defer func() {
			_ = conn.Close()
			if connection == 1 {
				closed <- struct{}{}
			}
		}()
		if _, _, errRead := conn.ReadMessage(); errRead != nil {
			return
		}
		completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
		if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
			t.Errorf("write completed response: %v", errWrite)
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	exec.store = &codexWebsocketSessionStore{sessions: make(map[string]*codexWebsocketSession)}
	auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`)}
	opts := cliproxyexecutor.Options{
		SourceFormat:       sdktranslator.FromString("openai-response"),
		ResponseFormat:     sdktranslator.FromString("openai-response"),
		ExecutionLifecycle: rejectingExecutionLifecycle{},
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: "nonstream-bind-failed",
		},
	}
	if _, errExecute := exec.Execute(context.Background(), auth, req, opts); errExecute == nil {
		t.Fatal("Execute() error = nil, want lifecycle bind failure")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("nonstream lifecycle bind failure did not close the upstream websocket")
	}
	sess := exec.getOrCreateSession("nonstream-bind-failed")
	sess.connMu.Lock()
	connected := sess.conn != nil
	sess.connMu.Unlock()
	if connected {
		t.Fatal("nonstream lifecycle bind failure left the closed connection attached to the session")
	}

	opts.ExecutionLifecycle = nil
	if _, errExecute := exec.Execute(context.Background(), auth, req, opts); errExecute != nil {
		t.Fatalf("second Execute() error = %v", errExecute)
	}
	if got := connections.Load(); got != 2 {
		t.Fatalf("websocket connections = %d, want 2 after bind failure", got)
	}
}

func TestCodexWebsocketLifecycleBindFailureReleasesSessionRequestLock(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	closed := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Errorf("upgrade websocket: %v", errUpgrade)
			return
		}
		defer func() {
			_ = conn.Close()
			closed <- struct{}{}
		}()
		for {
			if _, _, errRead := conn.ReadMessage(); errRead != nil {
				return
			}
		}
	}))
	defer server.Close()

	exec := NewCodexWebsocketsExecutor(&config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}})
	exec.store = &codexWebsocketSessionStore{sessions: make(map[string]*codexWebsocketSession)}
	auth := &cliproxyauth.Auth{ID: "auth-a", Provider: "codex", Attributes: map[string]string{"api_key": "sk-test", "base_url": server.URL}}
	req := cliproxyexecutor.Request{Model: "gpt-5-codex", Payload: []byte(`{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":"hello"}]}`)}
	opts := cliproxyexecutor.Options{
		SourceFormat:       sdktranslator.FromString("openai-response"),
		ResponseFormat:     sdktranslator.FromString("openai-response"),
		ExecutionLifecycle: rejectingExecutionLifecycle{},
		Metadata: map[string]any{
			cliproxyexecutor.ExecutionSessionMetadataKey: "bind-failed",
		},
	}
	if _, errExecute := exec.ExecuteStream(context.Background(), auth, req, opts); errExecute == nil {
		t.Fatal("ExecuteStream() error = nil, want lifecycle bind failure")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("lifecycle bind failure did not close the upstream websocket")
	}

	sess := exec.getOrCreateSession("bind-failed")
	acquired := make(chan struct{})
	go func() {
		sess.reqMu.Lock()
		close(acquired)
		sess.reqMu.Unlock()
	}()
	select {
	case <-acquired:
	case <-time.After(time.Second):
		t.Fatal("lifecycle bind failure left the session request lock held")
	}
}
