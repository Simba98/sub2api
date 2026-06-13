package service

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
)

// ForwardAsChatCompletions serves OpenAI Chat Completions clients through
// Antigravity accounts by reusing the Chat -> Responses -> Anthropic bridge,
// then forwarding through Antigravity's Gemini/v1internal upstream path.
func (s *AntigravityGatewayService) ForwardAsChatCompletions(ctx context.Context, c *gin.Context, account *Account, body []byte, isStickySession bool) (*ForwardResult, error) {
	startTime := time.Now()

	var ccReq apicompat.ChatCompletionsRequest
	if err := json.Unmarshal(body, &ccReq); err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
	}
	if strings.TrimSpace(ccReq.Model) == "" {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", "model is required")
	}

	originalModel := ccReq.Model
	clientStream := ccReq.Stream
	includeUsage := ccReq.StreamOptions != nil && ccReq.StreamOptions.IncludeUsage

	responsesReq, err := apicompat.ChatCompletionsToResponses(&ccReq)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
	}
	anthropicReq, err := apicompat.ResponsesToAnthropicRequest(responsesReq)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
	}
	anthropicReq.Stream = true

	claudeBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("marshal antigravity chat completions compat request: %w", err)
	}

	var claudeReq antigravity.ClaudeRequest
	if err := json.Unmarshal(claudeBody, &claudeReq); err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
	}

	mappedModel := s.getMappedModel(account, claudeReq.Model)
	if mappedModel == "" {
		MarkOpsClientBusinessLimited(c, OpsClientBusinessLimitedReasonLocalFeatureGate)
		return nil, writeAntigravityCCError(c, http.StatusForbidden, "permission_error", fmt.Sprintf("model %s not in whitelist", claudeReq.Model))
	}
	thinkingEnabled := claudeReq.Thinking != nil && (claudeReq.Thinking.Type == "enabled" || claudeReq.Thinking.Type == "adaptive")
	mappedModel = applyThinkingModelSuffix(mappedModel, thinkingEnabled)
	billingModel := mappedModel

	if s.tokenProvider == nil {
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Antigravity token provider not configured")
	}
	accessToken, err := s.tokenProvider.GetAccessToken(ctx, account)
	if err != nil {
		return nil, &UpstreamFailoverError{
			StatusCode:   http.StatusBadGateway,
			ResponseBody: []byte(`{"error":{"type":"authentication_error","message":"Failed to get upstream access token"},"type":"error"}`),
		}
	}

	projectID, err := resolveAntigravityProjectID(account)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
	}

	transformOpts := s.getClaudeTransformOptions(ctx)
	transformOpts.EnableIdentityPatch = true
	geminiBody, err := antigravity.TransformClaudeToGeminiWithOptions(&claudeReq, projectID, mappedModel, transformOpts)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadRequest, "invalid_request_error", "Invalid request")
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	prefix := logPrefix(getSessionID(c), account.Name)
	result, err := s.antigravityRetryLoop(antigravityRetryLoopParams{
		ctx:             ctx,
		prefix:          prefix,
		account:         account,
		proxyURL:        proxyURL,
		accessToken:     accessToken,
		action:          "streamGenerateContent",
		body:            geminiBody,
		c:               c,
		httpUpstream:    s.httpUpstream,
		settingService:  s.settingService,
		accountRepo:     s.accountRepo,
		handleError:     s.handleUpstreamError,
		requestedModel:  originalModel,
		isStickySession: isStickySession,
	})
	if err != nil {
		if switchErr, ok := IsAntigravityAccountSwitchError(err); ok {
			return nil, &UpstreamFailoverError{StatusCode: http.StatusServiceUnavailable, ForceCacheBilling: switchErr.IsStickySession}
		}
		if c != nil && c.Request != nil && c.Request.Context().Err() != nil {
			return nil, writeAntigravityCCError(c, http.StatusBadGateway, "client_disconnected", "Client disconnected before upstream response")
		}
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Upstream request failed after retries")
	}
	resp := result.resp
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody := s.readUpstreamErrorBody(resp)
		unwrappedBody, unwrapErr := s.unwrapV1InternalResponse(respBody)
		if unwrapErr != nil || len(unwrappedBody) == 0 {
			unwrappedBody = respBody
		}
		s.handleUpstreamError(ctx, prefix, account, resp.StatusCode, resp.Header, respBody, originalModel, 0, "", isStickySession)
		upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractAntigravityErrorMessage(unwrappedBody)))
		upstreamDetail := s.getUpstreamErrorDetail(unwrappedBody)
		setOpsUpstreamError(c, resp.StatusCode, upstreamMsg, upstreamDetail)
		if s.shouldFailoverUpstreamError(resp.StatusCode) {
			return nil, &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: unwrappedBody}
		}
		return nil, writeAntigravityCCError(c, mapUpstreamStatusCode(resp.StatusCode), "server_error", upstreamMsg)
	}

	requestID := resp.Header.Get("x-request-id")
	if requestID != "" {
		c.Header("x-request-id", requestID)
	}

	var usage *ClaudeUsage
	var firstTokenMs *int
	if clientStream {
		streamRes, err := s.handleChatCompletionsStreamingResponse(c, resp, startTime, originalModel, includeUsage)
		if err != nil {
			return nil, err
		}
		usage = streamRes.usage
		firstTokenMs = streamRes.firstTokenMs
	} else {
		streamRes, err := s.handleChatCompletionsStreamToNonStreaming(c, resp, originalModel)
		if err != nil {
			return nil, err
		}
		usage = streamRes.usage
	}
	if usage == nil {
		usage = &ClaudeUsage{}
	}

	return &ForwardResult{
		RequestID:       requestID,
		Usage:           *usage,
		Model:           originalModel,
		UpstreamModel:   billingModel,
		Stream:          clientStream,
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
		ReasoningEffort: extractCCReasoningEffortFromBody(body),
	}, nil
}

func writeAntigravityCCError(c *gin.Context, statusCode int, errType, message string) error {
	writeGatewayCCError(c, statusCode, errType, message)
	if strings.TrimSpace(message) == "" {
		message = http.StatusText(statusCode)
	}
	return fmt.Errorf("antigravity chat completions error: status=%d type=%s message=%s", statusCode, errType, message)
}

func (s *AntigravityGatewayService) handleChatCompletionsStreamToNonStreaming(c *gin.Context, resp *http.Response, originalModel string) (*geminiStreamResult, error) {
	collected, usage, err := s.collectChatCompletionsGeminiSSE(resp.Body)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Failed to read upstream stream")
	}
	collectedBytes, _ := json.Marshal(collected)
	chatResp, usage2, err := geminiResponseToChatCompletions(collected, originalModel, collectedBytes, usage)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Failed to parse upstream response")
	}
	c.JSON(http.StatusOK, chatResp)
	return &geminiStreamResult{usage: usage2}, nil
}

func (s *AntigravityGatewayService) collectChatCompletionsGeminiSSE(r io.Reader) (map[string]any, *ClaudeUsage, error) {
	reader := bufio.NewReader(r)
	var last map[string]any
	var usage *ClaudeUsage
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(trimmed, "data:") {
				payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				if payload != "" && payload != "[DONE]" {
					rawBytes := []byte(payload)
					if innerBytes, uwErr := s.unwrapV1InternalResponse(rawBytes); uwErr == nil {
						rawBytes = innerBytes
					}
					var geminiResp map[string]any
					if err := json.Unmarshal(rawBytes, &geminiResp); err == nil {
						last = geminiResp
						if u := extractGeminiUsage(rawBytes); u != nil {
							usage = u
						}
					}
				}
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, usage, err
		}
	}
	if last == nil {
		last = map[string]any{}
	}
	return last, usage, nil
}

func (s *AntigravityGatewayService) handleChatCompletionsStreamingResponse(c *gin.Context, resp *http.Response, startTime time.Time, originalModel string, includeUsage bool) (*geminiStreamResult, error) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	collected, usage, err := s.collectChatCompletionsGeminiSSE(resp.Body)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Failed to read upstream stream")
	}
	collectedBytes, _ := json.Marshal(collected)
	chatResp, usage2, err := geminiResponseToChatCompletions(collected, originalModel, collectedBytes, usage)
	if err != nil {
		return nil, writeAntigravityCCError(c, http.StatusBadGateway, "upstream_error", "Failed to parse upstream response")
	}
	if usage2 != nil {
		usage = usage2
	}

	content := ""
	if len(chatResp.Choices) > 0 {
		content = chatMessageContentString(chatResp.Choices[0].Message.Content)
	}
	chunk := apicompat.ChatCompletionsChunk{
		ID:      chatResp.ID,
		Object:  "chat.completion.chunk",
		Created: chatResp.Created,
		Model:   originalModel,
		Choices: []apicompat.ChatChunkChoice{{
			Index: 0,
			Delta: apicompat.ChatDelta{Role: "assistant", Content: &content},
		}},
	}

	if sse, err := apicompat.ChatChunkToSSE(chunk); err == nil {
		_, _ = io.WriteString(c.Writer, sse)
		flusher.Flush()
	}
	finishReason := "stop"
	if len(chatResp.Choices) > 0 && strings.TrimSpace(chatResp.Choices[0].FinishReason) != "" {
		finishReason = chatResp.Choices[0].FinishReason
	}
	finalChunk := apicompat.ChatCompletionsChunk{
		ID:      chatResp.ID,
		Object:  "chat.completion.chunk",
		Created: chatResp.Created,
		Model:   originalModel,
		Choices: []apicompat.ChatChunkChoice{{Index: 0, Delta: apicompat.ChatDelta{}, FinishReason: &finishReason}},
	}
	if includeUsage && usage != nil {
		finalChunk.Usage = &apicompat.ChatUsage{PromptTokens: usage.InputTokens, CompletionTokens: usage.OutputTokens, TotalTokens: usage.InputTokens + usage.OutputTokens}
	}
	if sse, err := apicompat.ChatChunkToSSE(finalChunk); err == nil {
		_, _ = io.WriteString(c.Writer, sse)
	}
	_, _ = io.WriteString(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
	firstTokenMs := int(time.Since(startTime).Milliseconds())
	return &geminiStreamResult{usage: usage, firstTokenMs: &firstTokenMs}, nil
}

func chatMessageContentString(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	return string(raw)
}
