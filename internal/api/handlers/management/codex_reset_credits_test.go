package management

import (
	"encoding/json"
	"testing"
)

func TestCodexResetCreditConsumeBodyUsesRedeemRequestID(t *testing.T) {
	body, err := codexResetCreditConsumeBody("request-123", "")
	if err != nil {
		t.Fatalf("consume body: %v", err)
	}

	var parsed map[string]string
	if errUnmarshal := json.Unmarshal(body, &parsed); errUnmarshal != nil {
		t.Fatalf("unmarshal consume body: %v", errUnmarshal)
	}
	if parsed["redeem_request_id"] != "request-123" {
		t.Fatalf("redeem_request_id = %q, want request-123", parsed["redeem_request_id"])
	}
	if _, ok := parsed["credit_id"]; ok {
		t.Fatalf("consume body unexpectedly contained credit_id: %s", string(body))
	}
}

func TestCodexResetCreditConsumeBodyIncludesSelectedCreditID(t *testing.T) {
	body, err := codexResetCreditConsumeBody("request-123", "credit-456")
	if err != nil {
		t.Fatalf("consume body: %v", err)
	}

	var parsed map[string]string
	if errUnmarshal := json.Unmarshal(body, &parsed); errUnmarshal != nil {
		t.Fatalf("unmarshal consume body: %v", errUnmarshal)
	}
	if parsed["redeem_request_id"] != "request-123" {
		t.Fatalf("redeem_request_id = %q, want request-123", parsed["redeem_request_id"])
	}
	if parsed["credit_id"] != "credit-456" {
		t.Fatalf("credit_id = %q, want credit-456", parsed["credit_id"])
	}
}
