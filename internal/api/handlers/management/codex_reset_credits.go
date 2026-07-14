package management

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
)

// Codex "rate limit reset credit" endpoints proxy to ChatGPT's backend on
// behalf of a stored Codex account. They power the dashboard's refresh +
// redeem buttons: OpenAI grants one free rate-limit reset per month, and these
// endpoints let an operator inspect the granted credit and redeem it.
const (
	codexResetCreditsURL        = "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
	codexResetCreditsConsumeURL = "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume"
	codexWebReferer             = "https://chatgpt.com/"
	codexWebOriginator          = "codex_cli_rs"
	codexCLIUserAgent           = "codex-cli"
)

// GetCodexResetCredits fetches the available rate-limit reset credits for a
// Codex account selected by auth_index.
//
// Endpoint:
//
//	GET /v0/management/codex-reset-credits?auth_index=<index>
func (h *Handler) GetCodexResetCredits(c *gin.Context) {
	authIndex := firstNonEmptyQuery(c, "auth_index", "authIndex", "AuthIndex")
	if authIndex == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "auth_index is required"})
		return
	}
	auth := h.authByIndex(authIndex)
	if auth == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "auth file not found"})
		return
	}

	status, respBody, err := h.codexBackendRequest(c.Request.Context(), auth, http.MethodGet, codexResetCreditsURL, nil)
	if err != nil {
		log.WithError(err).Errorf("codex reset-credits fetch failed: auth_index=%s", authIndex)
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	if status < http.StatusOK || status >= http.StatusMultipleChoices {
		log.Errorf("codex reset-credits fetch upstream error: auth_index=%s status=%d body=%s", authIndex, status, codexTruncateBody(respBody))
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("upstream status %d", status), "body": string(respBody)})
		return
	}

	// Pass the upstream JSON through unchanged so the dashboard can read
	// credits[] and available_count directly.
	c.Data(http.StatusOK, "application/json", respBody)
}

type consumeCodexResetRequest struct {
	AuthIndexSnake  *string `json:"auth_index"`
	AuthIndexCamel  *string `json:"authIndex"`
	AuthIndexPascal *string `json:"AuthIndex"`
	// CreditID is optional; when empty upstream chooses an available credit.
	CreditID string `json:"credit_id"`
}

// ConsumeCodexResetCredit redeems one rate-limit reset credit for a Codex
// account. When no credit_id is supplied upstream chooses an available credit.
//
// Endpoint:
//
//	POST /v0/management/codex-reset-credits/consume
//	body: {"auth_index":"<index>","credit_id":"<optional>"}
func (h *Handler) ConsumeCodexResetCredit(c *gin.Context) {
	var body consumeCodexResetRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
		return
	}
	authIndex := firstNonEmptyString(body.AuthIndexSnake, body.AuthIndexCamel, body.AuthIndexPascal)
	if authIndex == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "auth_index is required"})
		return
	}
	auth := h.authByIndex(authIndex)
	if auth == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "auth file not found"})
		return
	}

	ctx := c.Request.Context()
	creditID := strings.TrimSpace(body.CreditID)

	redeemRequestID := uuid.NewString()
	reqBody, err := codexResetCreditConsumeBody(redeemRequestID, creditID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to build request body"})
		return
	}

	status, respBody, err := h.codexBackendRequest(ctx, auth, http.MethodPost, codexResetCreditsConsumeURL, reqBody)
	if err != nil {
		log.WithError(err).Errorf("codex reset-credits consume failed: auth_index=%s credit_id=%s", authIndex, creditID)
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	if status < http.StatusOK || status >= http.StatusMultipleChoices {
		log.Errorf("codex reset-credits consume upstream error: auth_index=%s credit_id=%s status=%d body=%s", authIndex, creditID, status, codexTruncateBody(respBody))
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("upstream status %d", status), "body": string(respBody)})
		return
	}

	log.Infof("codex reset-credits consumed: auth_index=%s credit_id=%s", authIndex, creditID)
	c.Data(http.StatusOK, "application/json", respBody)
}

// codexBackendRequest performs an authenticated call to ChatGPT's backend for
// the given Codex account. It resolves (and refreshes if needed) the account's
// access token, sets the ChatGPT web headers, and routes through the account's
// configured proxy. It returns the upstream status code and raw body.
func (h *Handler) codexBackendRequest(ctx context.Context, auth *coreauth.Auth, method, urlStr string, body []byte) (int, []byte, error) {
	token, errToken := h.resolveTokenForAuth(ctx, auth)
	if errToken != nil {
		return 0, nil, fmt.Errorf("resolve access token: %w", errToken)
	}
	if strings.TrimSpace(token) == "" {
		return 0, nil, fmt.Errorf("access token not found for account")
	}

	var reqReader io.Reader
	if body != nil {
		reqReader = bytes.NewReader(body)
	}
	req, errReq := http.NewRequestWithContext(ctx, method, urlStr, reqReader)
	if errReq != nil {
		return 0, nil, fmt.Errorf("build request: %w", errReq)
	}

	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(token))
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Referer", codexWebReferer)
	req.Header.Set("Originator", codexWebOriginator)
	req.Header.Set("User-Agent", codexCLIUserAgent)
	if targetPath := strings.TrimPrefix(urlStr, "https://chatgpt.com"); targetPath != urlStr {
		req.Header.Set("x-openai-target-path", targetPath)
		req.Header.Set("x-openai-target-route", targetPath)
	}
	if accountID := codexAccountID(auth); accountID != "" {
		req.Header.Set("ChatGPT-Account-Id", accountID)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	// No client timeout here: per project convention, timeouts are only allowed
	// during credential acquisition (handled in resolveTokenForAuth). The
	// request context cancels the call when the caller disconnects.
	httpClient := &http.Client{Transport: h.apiCallTransport(auth)}
	resp, errDo := httpClient.Do(req)
	if errDo != nil {
		return 0, nil, fmt.Errorf("request failed: %w", errDo)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("response body close error: %v", errClose)
		}
	}()

	respBody, errRead := io.ReadAll(resp.Body)
	if errRead != nil {
		return resp.StatusCode, nil, fmt.Errorf("read response: %w", errRead)
	}
	return resp.StatusCode, respBody, nil
}

func codexResetCreditConsumeBody(redeemRequestID, creditID string) ([]byte, error) {
	payload := map[string]string{"redeem_request_id": redeemRequestID}
	if strings.TrimSpace(creditID) != "" {
		payload["credit_id"] = strings.TrimSpace(creditID)
	}
	return json.Marshal(payload)
}

// codexAccountID extracts the ChatGPT account id stored on a Codex auth.
func codexAccountID(auth *coreauth.Auth) string {
	if auth == nil || auth.Metadata == nil {
		return ""
	}
	for _, key := range []string{"account_id", "chatgpt_account_id"} {
		if v, ok := auth.Metadata[key].(string); ok {
			if s := strings.TrimSpace(v); s != "" {
				return s
			}
		}
	}
	return ""
}

// codexTruncateBody bounds upstream bodies included in logs so we keep enough
// to investigate failures without dumping large payloads.
func codexTruncateBody(body []byte) string {
	const maxLen = 512
	s := strings.TrimSpace(string(body))
	if len(s) > maxLen {
		return s[:maxLen] + "...(truncated)"
	}
	return s
}

// firstNonEmptyQuery returns the first non-empty query value among keys.
func firstNonEmptyQuery(c *gin.Context, keys ...string) string {
	for _, key := range keys {
		if v := strings.TrimSpace(c.Query(key)); v != "" {
			return v
		}
	}
	return ""
}
