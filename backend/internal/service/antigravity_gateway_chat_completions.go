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
	collected, usage, err := collectGeminiSSE(resp.Body, true)
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

	anthState := apicompat.NewAnthropicEventToResponsesState()
	anthState.Model = originalModel
	ccState := apicompat.NewResponsesEventToChatState()
	ccState.Model = originalModel
	ccState.IncludeUsage = includeUsage

	var usage ClaudeUsage
	var firstTokenMs *int
	firstChunk := true

	writeChatChunk := func(chunk apicompat.ChatCompletionsChunk) bool {
		sse, err := apicompat.ChatChunkToSSE(chunk)
		if err != nil {
			return false
		}
		if _, err := io.WriteString(c.Writer, sse); err != nil {
			return true
		}
		return false
	}

	emitAnthropicEvent := func(evt *apicompat.AnthropicStreamEvent) bool {
		responsesEvents := apicompat.AnthropicEventToResponsesEvents(evt, anthState)
		for _, resEvt := range responsesEvents {
			chunks := apicompat.ResponsesEventToChatChunks(&resEvt, ccState)
			for _, chunk := range chunks {
				if writeChatChunk(chunk) {
					return true
				}
			}
		}
		flusher.Flush()
		return false
	}

	messageID := "msg_" + randomHex(12)
	if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{
		Type: "message_start",
		Message: &apicompat.AnthropicResponse{
			ID:         messageID,
			Type:       "message",
			Role:       "assistant",
			Model:      originalModel,
			Content:    []apicompat.AnthropicContentBlock{},
			StopReason: nil,
			Usage:      apicompat.AnthropicUsage{},
		},
	}) {
		return &geminiStreamResult{usage: &usage}, nil
	}

	finishReason := ""
	sawToolUse := false
	nextBlockIndex := 0
	openBlockIndex := -1
	openBlockType := ""
	seenText := ""
	openToolIndex := -1
	openToolName := ""
	seenToolJSON := ""

	closeOpenBlock := func() bool {
		if openBlockIndex < 0 {
			return false
		}
		disconnected := emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_stop"})
		openBlockIndex = -1
		openBlockType = ""
		return disconnected
	}
	closeOpenTool := func() bool {
		if openToolIndex < 0 {
			return false
		}
		disconnected := emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_stop"})
		openToolIndex = -1
		openToolName = ""
		seenToolJSON = ""
		return disconnected
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if strings.HasPrefix(trimmed, "data:") {
				payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				if payload != "" && payload != "[DONE]" {
					rawBytes := []byte(payload)
					if innerBytes, unwrapErr := s.unwrapV1InternalResponse(rawBytes); unwrapErr == nil {
						rawBytes = innerBytes
					}
					var geminiResp map[string]any
					if json.Unmarshal(rawBytes, &geminiResp) == nil {
						if firstChunk {
							firstChunk = false
							ms := int(time.Since(startTime).Milliseconds())
							firstTokenMs = &ms
						}
						if reason := extractGeminiFinishReason(geminiResp); reason != "" {
							finishReason = reason
						}
						if currentUsage := extractGeminiUsage(rawBytes); currentUsage != nil {
							usage = *currentUsage
						}

						for _, part := range extractGeminiParts(geminiResp) {
							if text, ok := part["text"].(string); ok && text != "" {
								if openToolIndex >= 0 && closeOpenTool() {
									return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
								}
								delta, newSeen := computeGeminiTextDelta(seenText, text)
								seenText = newSeen
								if delta == "" {
									continue
								}
								if openBlockType != "text" {
									if closeOpenBlock() {
										return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
									}
									idx := nextBlockIndex
									nextBlockIndex++
									openBlockIndex = idx
									openBlockType = "text"
									if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_start", Index: &idx, ContentBlock: &apicompat.AnthropicContentBlock{Type: "text", Text: ""}}) {
										return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
									}
								}
								if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_delta", Delta: &apicompat.AnthropicDelta{Type: "text_delta", Text: delta}}) {
									return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
								}
								continue
							}

							if functionCall, ok := part["functionCall"].(map[string]any); ok && functionCall != nil {
								name, _ := functionCall["name"].(string)
								if strings.TrimSpace(name) == "" {
									name = "tool"
								}
								if closeOpenBlock() {
									return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
								}
								if openToolIndex >= 0 && openToolName != name && closeOpenTool() {
									return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
								}
								if openToolIndex < 0 {
									idx := nextBlockIndex
									nextBlockIndex++
									openToolIndex = idx
									openToolName = name
									sawToolUse = true
									if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_start", Index: &idx, ContentBlock: &apicompat.AnthropicContentBlock{Type: "tool_use", ID: "toolu_" + randomHex(8), Name: name, Input: json.RawMessage(`{}`)}}) {
										return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
									}
								}
								argsJSON := "{}"
								switch value := functionCall["args"].(type) {
								case string:
									if strings.TrimSpace(value) != "" {
										argsJSON = value
									}
								case nil:
								default:
									if encoded, marshalErr := json.Marshal(value); marshalErr == nil && len(encoded) > 0 {
										argsJSON = string(encoded)
									}
								}
								delta, newSeen := computeGeminiTextDelta(seenToolJSON, argsJSON)
								seenToolJSON = newSeen
								if delta != "" && emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "content_block_delta", Delta: &apicompat.AnthropicDelta{Type: "input_json_delta", PartialJSON: delta}}) {
									return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
								}
							}
						}
					}
				}
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stream read error: %w", err)
		}
	}

	if closeOpenBlock() || closeOpenTool() {
		return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
	}
	stopReason := mapGeminiFinishReasonToClaudeStopReason(finishReason)
	if sawToolUse {
		stopReason = "tool_use"
	}
	anthState.InputTokens = usage.InputTokens
	anthState.CacheReadInputTokens = usage.CacheReadInputTokens
	if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "message_delta", Delta: &apicompat.AnthropicDelta{Type: "message_delta", StopReason: stopReason}, Usage: &apicompat.AnthropicUsage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens, CacheReadInputTokens: usage.CacheReadInputTokens}}) {
		return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
	}
	if emitAnthropicEvent(&apicompat.AnthropicStreamEvent{Type: "message_stop"}) {
		return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
	}
	for _, responseEvent := range apicompat.FinalizeAnthropicResponsesStream(anthState) {
		for _, chunk := range apicompat.ResponsesEventToChatChunks(&responseEvent, ccState) {
			if writeChatChunk(chunk) {
				return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
			}
		}
	}
	for _, chunk := range apicompat.FinalizeResponsesChatStream(ccState) {
		if writeChatChunk(chunk) {
			return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
		}
	}
	_, _ = io.WriteString(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
	return &geminiStreamResult{usage: &usage, firstTokenMs: firstTokenMs}, nil
}
