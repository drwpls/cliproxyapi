package management

import (
	"encoding/json"
	"testing"
	"time"
)

func TestCodexResetCreditConsumeBodyUsesRedeemRequestID(t *testing.T) {
	body, err := codexResetCreditConsumeBody("credit-123")
	if err != nil {
		t.Fatalf("consume body: %v", err)
	}

	var parsed map[string]string
	if errUnmarshal := json.Unmarshal(body, &parsed); errUnmarshal != nil {
		t.Fatalf("unmarshal consume body: %v", errUnmarshal)
	}
	if parsed["redeem_request_id"] != "credit-123" {
		t.Fatalf("redeem_request_id = %q, want credit-123", parsed["redeem_request_id"])
	}
	if _, ok := parsed["credit_id"]; ok {
		t.Fatalf("consume body unexpectedly contained credit_id: %s", string(body))
	}
}

func TestMostRecentExpiredCreditIDPicksExpiredClosestToNow(t *testing.T) {
	now := time.Date(2026, 6, 22, 12, 0, 0, 0, time.UTC)
	body := []byte(`{
		"credits": [
			{"id": "old-expired", "redeem_request_id": "redeem-old", "expires_at": "2026-06-21T12:00:00Z"},
			{"id": "future", "redeem_request_id": "redeem-future", "expires_at": "2026-06-23T12:00:00Z"},
			{"id": "recent-expired", "redeem_request_id": "redeem-recent", "expires_at": "2026-06-22T11:59:00Z"}
		]
	}`)

	id, ok := mostRecentExpiredCreditID(body, now)
	if !ok {
		t.Fatal("expected expired credit")
	}
	if id != "redeem-recent" {
		t.Fatalf("credit id = %q, want redeem-recent", id)
	}
}

func TestMostRecentExpiredCreditIDFallsBackToID(t *testing.T) {
	now := time.Date(2026, 6, 22, 12, 0, 0, 0, time.UTC)
	body := []byte(`{"credits":[{"id":"fallback-id","expires_at":"2026-06-22T11:59:00Z"}]}`)

	id, ok := mostRecentExpiredCreditID(body, now)
	if !ok {
		t.Fatal("expected expired credit")
	}
	if id != "fallback-id" {
		t.Fatalf("credit id = %q, want fallback-id", id)
	}
}

func TestMostRecentExpiredCreditIDRejectsFutureOnly(t *testing.T) {
	now := time.Date(2026, 6, 22, 12, 0, 0, 0, time.UTC)
	body := []byte(`{"credits":[{"id":"future","expires_at":"2026-06-22T12:01:00Z"}]}`)

	if id, ok := mostRecentExpiredCreditID(body, now); ok {
		t.Fatalf("expected no expired credit, got %q", id)
	}
}
