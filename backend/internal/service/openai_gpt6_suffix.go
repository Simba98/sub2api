package service

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Match only the two public Astra names, never an unrelated GPT-6 family.
func gpt6ReasoningSuffix(model string) string {
	normalized := canonicalizeOpenAIModelAliasSpelling(model)
	for _, base := range []string{"gpt-6-astra", "gpt-6"} {
		if suffix, ok := strings.CutPrefix(normalized, base+"-"); ok {
			switch suffix {
			case "low", "medium", "high", "xhigh", "max":
				return suffix
			}
		}
	}
	return ""
}

// Resolve the suffix before passthrough and native forwarding diverge. Explicit
// effort wins; other reasoning fields (including Astra's mode) stay intact.
func normalizeGPT6ResponsesReasoningSuffix(body []byte) ([]byte, error) {
	effort := gpt6ReasoningSuffix(gjson.GetBytes(body, "model").String())
	if effort == "" {
		return body, nil
	}
	normalized := body
	var err error
	if !gjson.GetBytes(body, "reasoning.effort").Exists() {
		if explicit := gjson.GetBytes(body, "reasoning_effort"); explicit.Type == gjson.String {
			effort = explicit.String()
		}
		normalized, err = sjson.SetBytes(normalized, "reasoning.effort", effort)
		if err != nil {
			return nil, fmt.Errorf("normalize GPT-6 suffix effort: %w", err)
		}
	}
	return normalized, nil
}

func hasOpenAIResponsesSystemMessage(body []byte) bool {
	input := gjson.GetBytes(body, "input")
	if input.IsObject() {
		return input.Get("role").String() == "system"
	}
	found := false
	if input.IsArray() {
		input.ForEach(func(_, item gjson.Result) bool {
			found = item.IsObject() && item.Get("role").String() == "system"
			return !found
		})
	}
	return found
}
