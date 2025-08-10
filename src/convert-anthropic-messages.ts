import {
  type LanguageModelV2CallWarning,
  type LanguageModelV2Message,
  type LanguageModelV2Prompt,
  type SharedV2ProviderOptions,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils";
import type {
  AnthropicAssistantMessage,
  AnthropicCacheControl,
  AnthropicMessage,
  AnthropicMessagesPrompt,
  AnthropicUserMessage,
  AnthropicToolResultContent,
} from "./anthropic-api-types";
import type { ModelMessage, FilePart, TextPart, ToolCallPart } from "ai";
import type { ReasoningUIPart } from 'ai';

export function convertToAnthropicMessagesPrompt({
  prompt,
  sendReasoning,
  warnings,
}: {
  prompt: LanguageModelV2Prompt;
  sendReasoning: boolean;
  warnings: LanguageModelV2CallWarning[];
}): {
  prompt: AnthropicMessagesPrompt;
  betas: Set<string>;
} {
  const betas = new Set<string>();
  const blocks = groupIntoBlocks(prompt);

  let system: AnthropicMessagesPrompt["system"] = undefined;
  const messages: AnthropicMessagesPrompt["messages"] = [];

  function getCacheControl(
    providerOptions: SharedV2ProviderOptions | undefined
  ): AnthropicCacheControl | undefined {
    const anthropic = providerOptions?.anthropic;

    // allow both cacheControl and cache_control:
    const cacheControlValue =
      anthropic?.cacheControl ?? anthropic?.cache_control;

    // Pass through value assuming it is of the correct type.
    // The Anthropic API will validate the value.
    return cacheControlValue as AnthropicCacheControl | undefined;
  }

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]!;
    const isLastBlock = i === blocks.length - 1;
    const type = block.type;

    switch (type) {
      case "system": {
        if (system != null) {
          throw new UnsupportedFunctionalityError({
            functionality:
              "Multiple system messages that are separated by user/assistant messages",
          });
        }

        system = block.messages.map(({ content, providerOptions }) => ({
          type: "text",
          text: content,
          cache_control: getCacheControl(providerOptions),
        }));

        break;
      }

      case "user": {
        // combines all user and tool messages in this block into a single message:
        const anthropicContent: AnthropicUserMessage["content"] = [];

        for (const message of block.messages) {
          if (message.role === "user") {
            const content = message.content;
            for (let j = 0; j < content.length; j++) {
              const part = content[j]!;

              const isLastPart = j === content.length - 1;
              const cacheControl =
                getCacheControl(part.providerOptions) ??
                (isLastPart ? getCacheControl(message.providerOptions) : undefined);

              if (part.type === "text") {
                anthropicContent.push({
                  type: "text",
                  text: part.text,
                  cache_control: cacheControl,
                });
              } else if (part.type === "file") {
                const mediaType = part.mediaType;
                if (mediaType === "application/pdf") {
                  betas.add("pdfs-2024-09-25");
                  anthropicContent.push({
                    type: "document",
                    source:
                      part.data instanceof URL
                        ? { type: "url", url: part.data.toString() }
                        : {
                          type: "base64",
                          media_type: "application/pdf",
                          data: typeof part.data === "string"
                            ? part.data
                            : convertUint8ArrayToBase64(part.data),
                        },
                    cache_control: cacheControl,
                  });
                } else if (mediaType?.startsWith("image/")) {
                  anthropicContent.push({
                    type: "image",
                    source:
                      part.data instanceof URL
                        ? { type: "url", url: part.data.toString() }
                        : {
                          type: "base64",
                          media_type: mediaType ?? "image/jpeg",
                          data: typeof part.data === "string"
                            ? part.data
                            : convertUint8ArrayToBase64(part.data),
                        },
                    cache_control: cacheControl,
                  });
                } else {
                  throw new UnsupportedFunctionalityError({
                    functionality: `Unsupported user file media type: ${mediaType}`,
                  });
                }
              }
            }
          } else if (message.role === "tool") {
            const content = message.content;
            for (let i = 0; i < content.length; i++) {
              const part = content[i]!;

              const isLastPart = i === content.length - 1;
              const cacheControl =
                getCacheControl(part.providerOptions) ??
                (isLastPart ? getCacheControl(message.providerOptions) : undefined);

              // Map LanguageModelV2ToolResultPart.output to Anthropic tool_result content
              let toolResultContent: AnthropicToolResultContent["content"];
              let isError: boolean | undefined;

              switch (part.output.type) {
                case "text":
                  toolResultContent = part.output.value;
                  isError = false;
                  break;
                case "json":
                  toolResultContent = JSON.stringify(part.output.value);
                  isError = false;
                  break;
                case "error-text":
                  toolResultContent = part.output.value;
                  isError = true;
                  break;
                case "error-json":
                  toolResultContent = JSON.stringify(part.output.value);
                  isError = true;
                  break;
                case "content":
                  toolResultContent = part.output.value.map((c) =>
                    c.type === "text"
                      ? { type: "text" as const, text: c.text, cache_control: undefined }
                      : c.mediaType === "application/pdf"
                        ? { type: "text" as const, text: "[document content omitted]", cache_control: undefined }
                        : {
                          type: "image" as const,
                          source: { type: "base64" as const, media_type: c.mediaType, data: c.data },
                          cache_control: undefined,
                        }
                  );
                  isError = false;
                  break;
              }

              anthropicContent.push({
                type: "tool_result",
                tool_use_id: part.toolCallId,
                content: toolResultContent,
                is_error: isError,
                cache_control: cacheControl,
              });
            }
          } else {
            throw new Error(`Unsupported role in user block`);
          }
        }

        messages.push({ role: "user", content: anthropicContent });

        break;
      }

      case "assistant": {
        // combines multiple assistant messages in this block into a single message:
        const anthropicContent: AnthropicAssistantMessage["content"] = [];

        for (let j = 0; j < block.messages.length; j++) {
          const message = block.messages[j]!;
          const isLastMessage = j === block.messages.length - 1;
          const { content } = message;

          for (let k = 0; k < content.length; k++) {
            const part = content[k]!;
            const isLastContentPart = k === content.length - 1;

            // cache control: first add cache control from part.
            // for the last part of a message,
            // check also if the message has cache control.
            const cacheControl =
              getCacheControl(part.providerOptions) ??
              (isLastContentPart
                ? getCacheControl(message.providerOptions)
                : undefined);

            switch (part.type) {
              case "text": {
                anthropicContent.push({
                  type: "text",
                  text:
                    // trim the last text part if it's the last message in the block
                    // because Anthropic does not allow trailing whitespace
                    // in pre-filled assistant responses
                    isLastBlock && isLastMessage && isLastContentPart
                      ? part.text.trim()
                      : part.text,

                  cache_control: cacheControl,
                });
                break;
              }

              case "reasoning": {
                if (sendReasoning) {
                  anthropicContent.push({
                    type: "thinking",
                    thinking: part.text,
                    cache_control: cacheControl,
                  });
                } else {
                  warnings.push({
                    type: "other",
                    message:
                      "sending reasoning content is disabled for this model",
                  });
                }
                break;
              }

              case "tool-call": {
                // Check if we already have a tool call with this ID
                const existingToolCall = anthropicContent.find(
                  (c) => c.type === "tool_use" && c.id === part.toolCallId
                );
                
                // Skip duplicate tool calls (OpenAI doesn't allow duplicate IDs)
                if (!existingToolCall) {
                  anthropicContent.push({
                    type: "tool_use",
                    id: part.toolCallId,
                    name: part.toolName,
                    input: part.input,
                    cache_control: cacheControl,
                  });
                }
                break;
              }
            }
          }
        }

        messages.push({ role: "assistant", content: anthropicContent });

        break;
      }

      default: {
        const _exhaustiveCheck: never = type;
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`);
      }
    }
  }

  return {
    prompt: { system, messages },
    betas,
  };
}

type SystemBlock = {
  type: "system";
  messages: Array<LanguageModelV2Message & { role: "system" }>;
};
type AssistantBlock = {
  type: "assistant";
  messages: Array<LanguageModelV2Message & { role: "assistant" }>;
};
type UserBlock = {
  type: "user";
  messages: Array<LanguageModelV2Message & { role: "user" | "tool" }>;
};

function groupIntoBlocks(
  prompt: LanguageModelV2Prompt
): Array<SystemBlock | AssistantBlock | UserBlock> {
  const blocks: Array<SystemBlock | AssistantBlock | UserBlock> = [];
  let currentBlock: SystemBlock | AssistantBlock | UserBlock | undefined =
    undefined;

  for (const message of prompt) {
    const { role } = message;
    switch (role) {
      case "system": {
        if (currentBlock?.type !== "system") {
          currentBlock = { type: "system", messages: [] };
          blocks.push(currentBlock);
        }

        currentBlock.messages.push(message);
        break;
      }
      case "assistant": {
        if (currentBlock?.type !== "assistant") {
          currentBlock = { type: "assistant", messages: [] };
          blocks.push(currentBlock);
        }

        currentBlock.messages.push(message);
        break;
      }
      case "user": {
        if (currentBlock?.type !== "user") {
          currentBlock = { type: "user", messages: [] };
          blocks.push(currentBlock);
        }

        currentBlock.messages.push(message);
        break;
      }
      case "tool": {
        if (currentBlock?.type !== "user") {
          currentBlock = { type: "user", messages: [] };
          blocks.push(currentBlock);
        }

        currentBlock.messages.push(message);
        break;
      }
      default: {
        const _exhaustiveCheck: never = role;
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
      }
    }
  }

  return blocks;
}

export function convertFromAnthropicMessages(
  messages: ReadonlyArray<AnthropicMessage>
) {
  const result: ModelMessage[] = [];
  let toolCalls: Record<string, ToolCallPart> = {};

  for (const message of messages) {
    const messageContent: (
      | TextPart
      | FilePart
      | ReasoningUIPart
      | ToolCallPart
    )[] = [];

    if (typeof message.content !== "string") {
      message.content.forEach((content) => {
        switch (content.type) {
          case "text": {
            messageContent.push({
              type: "text",
              text: content.text,
            });
            break;
          }
          case "tool_use": {
            messageContent.push({
              type: "tool-call",
              input: content.input,
              toolCallId: content.id,
              toolName: content.name,
            });
            toolCalls[content.id] = {
              type: "tool-call",
              input: content.input,
              toolCallId: content.id,
              toolName: content.name,
            } as ToolCallPart;
            break;
          }
          case "tool_result": {
            const toolCall = toolCalls[content.tool_use_id];
            if (!toolCall) {
              throw new Error("Tool call not found");
            }
            // Map Anthropic tool_result content to LanguageModelV2ToolResultOutput
            const output = ((): any => {
              const c = content.content;
              if (typeof c === "string") {
                return { type: "text", value: c } as const;
              }
              // array of text/image contents
              const mapped = c.map((p) => {
                if (p.type === "text") {
                  return { type: "text" as const, text: p.text };
                }
                // image content: prefer base64 source
                if (p.type === "image") {
                  if (p.source.type === "base64") {
                    return {
                      type: "media" as const,
                      data: p.source.data,
                      mediaType: p.source.media_type,
                    };
                  } else {
                    // URL fallback: encode as text with URL
                    return { type: "text" as const, text: p.source.url };
                  }
                }
                // default to text
                return { type: "text" as const, text: "" };
              });
              return { type: "content", value: mapped } as const;
            })();

            result.push({
              role: "tool",
              content: [
                {
                  output,
                  toolCallId: content.tool_use_id,
                  toolName: toolCall.toolName,
                  type: "tool-result",
                },
              ],
            });
            break;
          }
        }
      });
    } else {
      messageContent.push({
        type: "text",
        text: message.content as string,
      });
    }

    if (messageContent.length > 0) {
      result.push({
        role: message.role,
        content: messageContent,
      } as ModelMessage);
    }
  }
  return result;
}
