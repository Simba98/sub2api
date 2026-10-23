package admin

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestSetAPIKeyUsageWindows(t *testing.T) {
	gin.SetMode(gin.TestMode)
	future := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	stub := newStubAdminService()
	stub.getAccountResult = &service.Account{ID: 7, Type: service.AccountTypeAPIKey, Extra: map[string]any{"keep": true}}
	handler := NewAccountHandler(stub, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil)
	router := gin.New()
	router.PUT("/accounts/:id/usage-windows", handler.SetAPIKeyUsageWindows)

	recorder := httptest.NewRecorder()
	body := `{"five_hour_reset_at":"` + future + `","seven_day_reset_at":null}`
	request := httptest.NewRequest(http.MethodPut, "/accounts/7/usage-windows", bytes.NewBufferString(body))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, 1, stub.updateAccountExtraCalls)
	require.Equal(t, future, stub.lastAccountExtraUpdates[service.APIKeyFiveHourResetAtExtraKey])
	require.Contains(t, stub.lastAccountExtraUpdates, service.APIKeySevenDayResetAtExtraKey)
	require.Nil(t, stub.lastAccountExtraUpdates[service.APIKeySevenDayResetAtExtraKey])
	require.NotContains(t, stub.lastAccountExtraUpdates, "keep")
}

func TestSetAPIKeyUsageWindowsRejectsInvalidRequests(t *testing.T) {
	tests := []struct {
		name        string
		accountType string
		body        string
	}{
		{"non apikey", service.AccountTypeOAuth, `{"five_hour_reset_at":null}`},
		{"invalid timestamp", service.AccountTypeAPIKey, `{"five_hour_reset_at":"tomorrow"}`},
		{"past timestamp", service.AccountTypeAPIKey, `{"seven_day_reset_at":"2020-01-01T00:00:00Z"}`},
		{"no fields", service.AccountTypeAPIKey, `{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			stub := newStubAdminService()
			stub.getAccountResult = &service.Account{ID: 7, Type: tt.accountType}
			handler := NewAccountHandler(stub, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil)
			router := gin.New()
			router.PUT("/accounts/:id/usage-windows", handler.SetAPIKeyUsageWindows)
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPut, "/accounts/7/usage-windows", bytes.NewBufferString(tt.body))
			request.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(recorder, request)
			require.Equal(t, http.StatusBadRequest, recorder.Code)
			require.Zero(t, stub.updateAccountExtraCalls)
		})
	}
}
