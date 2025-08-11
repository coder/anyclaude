import { describe, expect, it } from "bun:test";
import { convertToAnthropicMessagesPrompt } from "./convert-anthropic-messages";
import type { LanguageModelV2Prompt } from "@ai-sdk/provider";

describe("convertToAnthropicMessagesPrompt", () => {
  describe("duplicate tool call filtering", () => {
    it("should filter out duplicate tool calls with the same ID", () => {
      const prompt: LanguageModelV2Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "TodoWrite",
              input: { todos: ["item1", "item2"] },
            },
            {
              type: "tool-call",
              toolCallId: "call_123", // Duplicate ID
              toolName: "TodoWrite",
              input: {}, // Empty input
            },
          ],
        },
      ];

      const result = convertToAnthropicMessagesPrompt({
        prompt,
        sendReasoning: false,
        warnings: [],
      });

      // Should only have one tool_use in the output
      const assistantMessage = result.prompt.messages[0];
      expect(assistantMessage?.role).toBe("assistant");
      
      if (assistantMessage?.role === "assistant") {
        const toolUses = assistantMessage.content.filter(
          (c) => c.type === "tool_use"
        );
        expect(toolUses).toHaveLength(1);
        expect(toolUses[0]?.id).toBe("call_123");
        expect(toolUses[0]?.name).toBe("TodoWrite");
        expect(toolUses[0]?.input).toEqual({ todos: ["item1", "item2"] });
      }
    });

    it("should keep tool calls with different IDs", () => {
      const prompt: LanguageModelV2Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "TodoWrite",
              input: { todos: ["item1"] },
            },
            {
              type: "tool-call",
              toolCallId: "call_456", // Different ID
              toolName: "Read",
              input: { file: "test.txt" },
            },
          ],
        },
      ];

      const result = convertToAnthropicMessagesPrompt({
        prompt,
        sendReasoning: false,
        warnings: [],
      });

      const assistantMessage = result.prompt.messages[0];
      expect(assistantMessage?.role).toBe("assistant");
      
      if (assistantMessage?.role === "assistant") {
        const toolUses = assistantMessage.content.filter(
          (c) => c.type === "tool_use"
        );
        expect(toolUses).toHaveLength(2);
        expect(toolUses[0]?.id).toBe("call_123");
        expect(toolUses[0]?.name).toBe("TodoWrite");
        expect(toolUses[1]?.id).toBe("call_456");
        expect(toolUses[1]?.name).toBe("Read");
      }
    });

    it("should handle mixed content with duplicate tool calls", () => {
      const prompt: LanguageModelV2Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "Let me help you with that.",
            },
            {
              type: "tool-call",
              toolCallId: "call_abc",
              toolName: "Search",
              input: { query: "test" },
            },
            {
              type: "text",
              text: "Processing...",
            },
            {
              type: "tool-call",
              toolCallId: "call_abc", // Duplicate ID
              toolName: "Search",
              input: { query: "different" },
            },
          ],
        },
      ];

      const result = convertToAnthropicMessagesPrompt({
        prompt,
        sendReasoning: false,
        warnings: [],
      });

      const assistantMessage = result.prompt.messages[0];
      expect(assistantMessage?.role).toBe("assistant");
      
      if (assistantMessage?.role === "assistant") {
        // Should have 2 text blocks and 1 tool_use (duplicate filtered)
        const textBlocks = assistantMessage.content.filter(
          (c) => c.type === "text"
        );
        const toolUses = assistantMessage.content.filter(
          (c) => c.type === "tool_use"
        );
        
        expect(textBlocks).toHaveLength(2);
        expect(toolUses).toHaveLength(1);
        expect(toolUses[0]?.id).toBe("call_abc");
        expect(toolUses[0]?.input).toEqual({ query: "test" }); // Should keep the first one
      }
    });

    it("should handle empty tool call arrays correctly", () => {
      const prompt: LanguageModelV2Prompt = [
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "No tools needed.",
            },
          ],
        },
      ];

      const result = convertToAnthropicMessagesPrompt({
        prompt,
        sendReasoning: false,
        warnings: [],
      });

      const assistantMessage = result.prompt.messages[0];
      expect(assistantMessage?.role).toBe("assistant");
      
      if (assistantMessage?.role === "assistant") {
        const toolUses = assistantMessage.content.filter(
          (c) => c.type === "tool_use"
        );
        expect(toolUses).toHaveLength(0);
      }
    });
  });
});