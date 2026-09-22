//go:build unit

package service

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestGPT6ReasoningSuffixForward(t *testing.T) {
	for _, base := range []string{"gpt-6", "gpt-6-astra"} {
		for _, effort := range []string{"low", "medium", "high", "xhigh", "max"} {
			for _, passthrough := range []bool{false, true} {
				for _, stream := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/passthrough=%t/stream=%t", base, effort, passthrough, stream), func(t *testing.T) {
						s := newAstraOAuthSetup(t, passthrough)
						s.upstream.resp = gpt6TestResponse()
						body := astraRequestBody(base+"-"+effort, stream, "pro", "")
						result, err := s.svc.Forward(context.Background(), s.c, s.account, body)
						require.NoError(t, err)
						require.Equal(t, "gpt-6-astra", gjson.GetBytes(s.upstream.lastBody, "model").String())
						require.Equal(t, effort, gjson.GetBytes(s.upstream.lastBody, "reasoning.effort").String())
						require.Equal(t, "pro", gjson.GetBytes(s.upstream.lastBody, "reasoning.mode").String())
						require.Equal(t, base+"-"+effort, result.Model)
						require.NotNil(t, result.ReasoningEffort)
						require.Equal(t, effort, *result.ReasoningEffort)
					})
				}
			}
		}
	}
}

func gpt6TestResponse() *http.Response {
	return &http.Response{StatusCode: http.StatusOK,
		Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:   io.NopCloser(strings.NewReader(codexCompletedSSE(`{"id":"resp_test","model":"gpt-6-astra","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`)))}
}

func TestGPT6SuffixAliasesAndExplicitEffort(t *testing.T) {
	for _, base := range []string{"gpt-6", "gpt-6-astra", "openai/gpt-6", "OPENAI/GPT-6_ASTRA"} {
		for _, effort := range []string{"low", "medium", "high", "xhigh", "max"} {
			model := base + "-" + effort
			require.Equal(t, "gpt-6-astra", normalizeCodexModel(model))
			require.Equal(t, "gpt-6-astra", NormalizeOpenAICompatRequestedModel(model))
			require.Equal(t, effort, deriveOpenAIReasoningEffortFromModel(model))
		}
	}
	for _, model := range []string{"gpt-6-other-high", "gpt-6-astra-pro-max", "gpt-6-ultra", "gpt-6-astra-unknown"} {
		require.Empty(t, gpt6ReasoningSuffix(model))
		require.Equal(t, model, normalizeCodexModel(model))
	}
	for _, explicit := range []string{`"reasoning":{"effort":"low","mode":"pro","summary":"auto"}`, `"reasoning_effort":"low","reasoning":{"mode":"pro","summary":"auto"}`} {
		body := []byte(`{"model":"gpt-6-max",` + explicit + `}`)
		got, err := normalizeGPT6ResponsesReasoningSuffix(body)
		require.NoError(t, err)
		require.Equal(t, "low", gjson.GetBytes(got, "reasoning.effort").String())
		require.Equal(t, "pro", gjson.GetBytes(got, "reasoning.mode").String())
		require.Equal(t, "auto", gjson.GetBytes(got, "reasoning.summary").String())
	}
}

func TestCodexSystemMessageRollsBackPassthrough(t *testing.T) {
	for _, model := range []string{"gpt-6-astra", "gpt-6-max", "gpt-5.3-codex"} {
		for _, input := range []string{
			`[{"role":"system","content":"Keep this guidance."},{"role":"user","content":"hello"}]`,
			`[{"role":"user","content":"hello"},{"type":"message","role":"system","content":[{"type":"input_text","text":"Keep this guidance."}]}]`,
			`{"role":"system","content":"Keep this guidance."}`,
		} {
			t.Run(model+"/"+input, func(t *testing.T) {
				s := newAstraOAuthSetup(t, true)
				body := []byte(`{"model":"` + model + `","stream":true,"instructions":"Existing instructions.","input":` + input + `}`)
				_, err := s.svc.forwardOpenAIPassthrough(context.Background(), s.c, s.account, body, body, model, false, nil, true, time.Now())
				var rollback *openAIPassthroughRollbackError
				require.ErrorAs(t, err, &rollback)
				require.Equal(t, "system_message", rollback.Reason)
				require.Nil(t, s.upstream.lastReq)
				s.upstream.resp = gpt6TestResponse()
				_, err = s.svc.Forward(context.Background(), s.c, s.account, body)
				require.NoError(t, err)
				if model == "gpt-6-max" {
					require.Equal(t, "gpt-6-astra", gjson.GetBytes(s.upstream.lastBody, "model").String())
					require.Equal(t, "max", gjson.GetBytes(s.upstream.lastBody, "reasoning.effort").String())
				}
				require.False(t, hasOpenAIResponsesSystemMessage(s.upstream.lastBody))
				require.Contains(t, gjson.GetBytes(s.upstream.lastBody, "instructions").String(), "Keep this guidance.")
				require.Contains(t, gjson.GetBytes(s.upstream.lastBody, "instructions").String(), "Existing instructions.")
			})
		}
	}
}

func TestResponsesSystemDetectionIgnoresNestedContent(t *testing.T) {
	for _, body := range []string{
		`{"input":"system"}`,
		`{"input":[{"role":"developer","content":"system"}]}`,
		`{"input":[{"role":"user","content":[{"role":"system"}]}]}`,
		`{"input":[{"type":"function_call_output","output":"{\"role\":\"system\"}"}]}`,
	} {
		require.False(t, hasOpenAIResponsesSystemMessage([]byte(body)))
	}
}
func TestAPIKeySystemMessageKeepsPassthrough(t *testing.T) {
	s := newAstraOAuthSetup(t, true)
	s.account.Type = AccountTypeAPIKey
	s.account.Credentials = map[string]any{"api_key": "test-key"}
	s.upstream.resp = gpt6TestResponse()
	body := []byte(`{"model":"gpt-6-astra","stream":true,"instructions":"Existing instructions.","input":[{"role":"system","content":"Keep this guidance."},{"role":"user","content":"hello"}]}`)
	_, err := s.svc.Forward(context.Background(), s.c, s.account, body)
	require.NoError(t, err)
	require.True(t, hasOpenAIResponsesSystemMessage(s.upstream.lastBody))
	require.Equal(t, "Existing instructions.", gjson.GetBytes(s.upstream.lastBody, "instructions").String())
}
