//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// Synthetic accounts share an ordinary OpenAI group; no provider traffic is sent.
func passthroughModelFilterAccounts(passthrough bool, oauthPriority int) []Account {
	return []Account{
		{
			ID: 1, Name: "bailian-compatible", Platform: PlatformOpenAI, Type: AccountTypeAPIKey,
			Status: StatusActive, Schedulable: true, Concurrency: 1, Priority: 1,
			GroupIDs: []int64{17}, RateMultiplier: f64p(1),
			Credentials: map[string]any{
				"api_key": "synthetic-key", "base_url": "https://bailian.example",
				"model_mapping": map[string]any{"qwen3.8-max": "qwen3.8-max"},
			},
			Extra: map[string]any{"openai_passthrough": passthrough},
		},
		{
			ID: 2, Name: "openai-oauth", Platform: PlatformOpenAI, Type: AccountTypeOAuth,
			Status: StatusActive, Schedulable: true, Concurrency: 1, Priority: oauthPriority,
			GroupIDs: []int64{17}, RateMultiplier: f64p(1),
			Credentials: map[string]any{
				"access_token":  "synthetic-token",
				"model_mapping": map[string]any{"gpt-6.1-sol": "gpt-6.1-sol"},
			},
		},
	}
}

func TestOpenAIPassthroughModelFilterEligibility(t *testing.T) {
	for _, accountType := range []string{AccountTypeAPIKey, AccountTypeOAuth} {
		t.Run(accountType, func(t *testing.T) {
			for _, tc := range []struct {
				name  string
				flag  any
				model string
				want  bool
			}{
				{"default ignores leftover mapping", nil, "gpt-6.1-sol", true},
				{"disabled ignores leftover mapping", false, "gpt-6.1-sol", true},
				{"invalid flag does not opt in", "true", "gpt-6.1-sol", true},
				{"opt in rejects unlisted GPT model", true, "gpt-6.1-sol", false},
				{"opt in accepts exact source key", true, "gpt-6-astra", true},
				{"opt in accepts wildcard source key", true, "gpt-6-astra-high", true},
				{"API key accepts other provider source key", true, "qwen3.8-max", accountType == AccountTypeAPIKey},
				{"mapping targets do not grant access", true, "gpt-upstream-only", false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					account := &Account{
						Platform: PlatformOpenAI, Type: accountType,
						Extra: map[string]any{"openai_passthrough": true, "openai_passthrough_model_filter": tc.flag},
						Credentials: map[string]any{"model_mapping": map[string]any{
							"gpt-6-astra": "gpt-upstream-only", "gpt-6-astra-*": "wildcard-target",
							"qwen3.8-max": "gpt-6-astra",
						}},
					}
					require.Equal(t, tc.want, account.IsModelSupported(tc.model))
					require.Equal(t, tc.want, (&GatewayService{}).isModelSupportedByAccount(account, tc.model))
				})
			}
			account := &Account{Platform: PlatformOpenAI, Type: accountType,
				Extra: map[string]any{"openai_passthrough": true, "openai_passthrough_model_filter": true}}
			require.True(t, account.IsModelSupported("gpt-6.1-sol"), "empty restrictions allow GPT models")
			require.Equal(t, accountType == AccountTypeAPIKey, account.IsModelSupported("qwen3.8-max"), "only API-key passthrough can serve non-GPT models")
		})
	}
}

func TestOpenAIPassthroughModelFilterMixedPoolSelection(t *testing.T) {
	groupID := int64(17)
	for _, mode := range []string{"legacy", "load-batch"} {
		t.Run(mode, func(t *testing.T) {
			for _, tc := range []struct {
				name     string
				filter   bool
				model    string
				excluded map[int64]struct{}
				wantID   int64
			}{
				{"default preserves passthrough", false, "gpt-6.1-sol", nil, 1},
				{"filter chooses GPT account", true, "gpt-6.1-sol", nil, 2},
				{"filter keeps Qwen account", true, "qwen3.8-max", nil, 1},
				{"failover never leaks GPT to Qwen", true, "gpt-6.1-sol", map[int64]struct{}{2: {}}, 0},
			} {
				t.Run(tc.name, func(t *testing.T) {
					accounts := passthroughModelFilterAccounts(true, 2)
					accounts[0].Extra["openai_passthrough_model_filter"] = tc.filter
					repo := schedulerTestOpenAIAccountRepo{accounts: accounts}
					cfg := &config.Config{}
					cfg.Gateway.Scheduling.LoadBatchEnabled = mode == "load-batch"
					svc := &OpenAIGatewayService{accountRepo: repo, cfg: cfg}
					var selected *Account
					var err error
					if mode == "legacy" {
						selected, err = svc.SelectAccountForModelWithExclusions(context.Background(), &groupID, "", tc.model, tc.excluded)
					} else {
						svc.concurrencyService = NewConcurrencyService(stubConcurrencyCache{})
						selection, selectErr := svc.SelectAccountWithLoadAwareness(context.Background(), &groupID, "", tc.model, tc.excluded)
						err = selectErr
						if selection != nil {
							selected = selection.Account
							if selection.ReleaseFunc != nil {
								defer selection.ReleaseFunc()
							}
						}
					}
					if tc.wantID == 0 {
						require.Error(t, err)
						require.Nil(t, selected)
						return
					}
					require.NoError(t, err)
					require.NotNil(t, selected)
					require.Equal(t, tc.wantID, selected.ID)
				})
			}
		})
	}
}

func TestOpenAIOAuthPassthroughMixedPoolSelection(t *testing.T) {
	groupID := int64(17)
	for _, mode := range []string{"legacy", "load-batch", "advanced"} {
		for _, tc := range []struct {
			name     string
			model    string
			sticky   bool
			excluded map[int64]struct{}
			wantID   int64
		}{
			{"foreign model skips preferred OAuth", "gemini-3.8-pro", false, nil, 1},
			{"foreign model rejects stale OAuth binding", "gemini-3.8-pro", true, nil, 1},
			{"failover cannot fall back to OAuth", "gemini-3.8-pro", true, map[int64]struct{}{1: {}}, 0},
			{"GPT still reaches OAuth", "gpt-6.1-sol", false, map[int64]struct{}{1: {}}, 2},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				accounts := passthroughModelFilterAccounts(true, 0)
				accounts[1].Extra = map[string]any{"openai_passthrough": true}
				accounts[1].Credentials["model_mapping"] = map[string]any{"*": "gpt-6.1-sol"}
				cfg := &config.Config{}
				cfg.Gateway.Scheduling.LoadBatchEnabled = mode == "load-batch"
				cache := &schedulerTestGatewayCache{sessionBindings: map[string]int64{}}
				session := ""
				if tc.sticky {
					session = "stale-oauth-binding"
					cache.sessionBindings[session] = 2
				}
				svc := &OpenAIGatewayService{
					accountRepo: schedulerTestOpenAIAccountRepo{accounts: accounts}, cfg: cfg, cache: cache,
					concurrencyService: NewConcurrencyService(schedulerTestConcurrencyCache{}),
				}
				var selected *Account
				var err error
				if mode == "legacy" {
					selected, err = svc.SelectAccountForModelWithExclusions(context.Background(), &groupID, session, tc.model, tc.excluded)
				} else {
					var selection *AccountSelectionResult
					if mode == "advanced" {
						svc.rateLimitService = newOpenAIAdvancedSchedulerRateLimitService("true")
						t.Cleanup(resetOpenAIAdvancedSchedulerSettingCacheForTest)
						selection, _, err = svc.SelectAccountWithScheduler(context.Background(), &groupID, "", session, tc.model, tc.excluded, OpenAIUpstreamTransportAny, false)
					} else {
						selection, err = svc.SelectAccountWithLoadAwareness(context.Background(), &groupID, session, tc.model, tc.excluded)
					}
					if selection != nil {
						selected = selection.Account
						if selection.ReleaseFunc != nil {
							defer selection.ReleaseFunc()
						}
					}
				}
				if tc.wantID == 0 {
					require.Error(t, err)
					require.Nil(t, selected)
					return
				}
				require.NoError(t, err)
				require.NotNil(t, selected)
				require.Equal(t, tc.wantID, selected.ID)
			})
		}
	}
}

func TestOpenAIPassthroughModelFilterPreservesForwardBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	accounts := passthroughModelFilterAccounts(true, 2)
	accounts[0].Extra["openai_passthrough_model_filter"] = true
	accounts[0].Credentials["model_mapping"] = map[string]any{"qwen3.8-max": "must-not-rewrite"}
	repo := schedulerTestOpenAIAccountRepo{accounts: accounts}
	body := []byte(`{ "model": "qwen3.8-max", "input":"synthetic query", "tools":[{"type":"web_search"}], "stream":false }`)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}`)),
	}}
	svc := &OpenAIGatewayService{accountRepo: repo, cfg: &config.Config{}, httpUpstream: upstream}
	groupID := int64(17)
	account, err := svc.SelectAccountForModelWithExclusions(context.Background(), &groupID, "", "qwen3.8-max", nil)
	require.NoError(t, err)
	require.Equal(t, int64(1), account.ID)
	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, body, upstream.lastBody, "whitelist filtering must not activate model rewriting")
	require.Equal(t, "https://bailian.example/v1/responses", upstream.lastReq.URL.String())
}

func TestOpenAIPassthroughModelFilterGroupCatalog(t *testing.T) {
	accounts := passthroughModelFilterAccounts(true, 2)
	accounts[0].Extra["openai_passthrough_model_filter"] = true
	svc := &GatewayService{accountRepo: schedulerTestOpenAIAccountRepo{accounts: accounts}}
	groupID := int64(17)
	models := svc.GetAvailableModels(context.Background(), &groupID, PlatformOpenAI)
	require.ElementsMatch(t, []string{"qwen3.8-max", "gpt-6.1-sol"}, models,
		"a restricted passthrough account must not contribute the unrestricted default catalog")
}
