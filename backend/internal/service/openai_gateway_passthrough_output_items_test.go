package service

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestHandleStreamingResponsePassthroughRestoresTerminalToolOutput(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const item = `{"type":"function_call","id":"fc_diag","call_id":"call_diag","name":"powershell","arguments":"{\"command\":\"Write-Output DIAG_STEP_TWO\"}","status":"completed","vendor_field":"preserved"}`
	for _, eventOnly := range []bool{false, true} {
		for _, terminal := range []string{"response.completed", "response.done", "response.incomplete"} {
			for _, output := range []string{`[]`, `null`, `[{"type":"message","id":"existing"}]`} {
				t.Run(fmt.Sprintf("eventOnly=%v/%s/%s", eventOnly, terminal, output), func(t *testing.T) {
					frame := func(kind, fields string) string {
						if eventOnly {
							return "event: " + kind + "\ndata: {" + fields + "}\n\n"
						}
						return passthroughSSEData(`{"type":"` + kind + `",` + fields + `}`)
					}
					body := frame("response.output_item.done", `"output_index":0,"item":`+item) +
						frame(terminal, `"response":{"id":"resp_diag","status":"completed","output":`+output+`,"usage":{"input_tokens":10,"output_tokens":5}}`)
					rec := httptest.NewRecorder()
					c, _ := gin.CreateTestContext(rec)
					c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
					resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(body))}
					svc := &OpenAIGatewayService{}
					_, err := svc.handleStreamingResponsePassthrough(context.Background(), resp, c, &Account{ID: 1}, time.Now(), "gpt-6-astra", "gpt-6-astra")
					require.NoError(t, err)
					var result gjson.Result
					for _, payload := range collectSSEDataPayloads(t, rec.Body.String()) {
						if gjson.Get(payload, "response.id").String() == "resp_diag" {
							result = gjson.Get(payload, "response.output")
						}
					}
					if strings.Contains(output, "existing") {
						require.JSONEq(t, output, result.Raw)
					} else {
						require.JSONEq(t, "["+item+"]", result.Raw)
					}
				})
			}
		}
	}
}
