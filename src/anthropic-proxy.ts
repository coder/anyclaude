import type { ProviderV2 } from "@ai-sdk/provider";
import { jsonSchema, streamText, type Tool } from "ai";
import * as http from "http";
import * as https from "https";
import type { AnthropicMessagesRequest } from "./anthropic-api-types";
import { mapAnthropicStopReason } from "./anthropic-api-types";
import {
  convertFromAnthropicMessages,
  convertToAnthropicMessagesPrompt,
} from "./convert-anthropic-messages";
import { convertToAnthropicStream } from "./convert-to-anthropic-stream";
import { convertToLanguageModelMessage } from "./convert-to-language-model-prompt";
import { providerizeSchema } from "./json-schema";
import { 
  writeDebugToTempFile, 
  logDebugError, 
  displayDebugStartup,
  isDebugEnabled,
  isVerboseDebugEnabled,
  queueErrorMessage
} from "./debug";
import { filterDuplicateToolCalls } from "./filter-duplicates";

export type CreateAnthropicProxyOptions = {
  providers: Record<string, ProviderV2>;
  port?: number;
};

// createAnthropicProxy creates a proxy server that accepts
// Anthropic Message API requests and proxies them through
// the appropriate provider - converting the results back
// to the Anthropic Message API format.
export const createAnthropicProxy = ({
  port,
  providers,
}: CreateAnthropicProxyOptions): string => {
  // Log debug status on startup
  displayDebugStartup();
  
  const proxy = http
    .createServer((req, res) => {
      if (!req.url) {
        res.writeHead(400, {
          "Content-Type": "application/json",
        });
        res.end(
          JSON.stringify({
            error: "No URL provided",
          })
        );
        return;
      }

      const proxyToAnthropic = (body?: AnthropicMessagesRequest) => {
        delete req.headers["host"];

        const requestBody = body ? JSON.stringify(body) : null;
        const chunks: Buffer[] = [];
        const responseChunks: Buffer[] = [];

        const proxy = https.request(
          {
            host: "api.anthropic.com",
            path: req.url,
            method: req.method,
            headers: req.headers,
          },
          (proxiedRes) => {
            const statusCode = proxiedRes.statusCode ?? 500;
            
            // Collect response data for debugging
            proxiedRes.on('data', (chunk) => {
              responseChunks.push(chunk);
            });

            proxiedRes.on('end', () => {
              // Write debug info to temp file for 4xx errors (except 429)
              if (statusCode >= 400 && statusCode < 500 && statusCode !== 429) {
                const requestBodyToLog = requestBody 
                  ? JSON.parse(requestBody)
                  : chunks.length > 0 
                    ? (() => {
                        try {
                          return JSON.parse(Buffer.concat(chunks).toString());
                        } catch {
                          return Buffer.concat(chunks).toString();
                        }
                      })()
                    : null;

                const responseBody = Buffer.concat(responseChunks).toString();
                const debugFile = writeDebugToTempFile(
                  statusCode,
                  {
                    method: req.method,
                    url: req.url,
                    headers: req.headers,
                    body: requestBodyToLog,
                  },
                  {
                    statusCode,
                    headers: proxiedRes.headers,
                    body: responseBody,
                  }
                );

                if (debugFile) {
                  logDebugError("HTTP", statusCode, debugFile);
                }
              }
            });

            res.writeHead(statusCode, proxiedRes.headers);
            proxiedRes.pipe(res, {
              end: true,
            });
          }
        );

        if (requestBody) {
          proxy.end(requestBody);
        } else {
          req.on('data', (chunk) => {
            chunks.push(chunk);
            proxy.write(chunk);
          });
          req.on('end', () => {
            proxy.end();
          });
        }
      };

      if (!req.url.startsWith("/v1/messages")) {
        proxyToAnthropic();
        return;
      }

      (async () => {
        const body = await new Promise<AnthropicMessagesRequest>(
          (resolve, reject) => {
            let body = "";
            req.on("data", (chunk) => {
              body += chunk;
            });
            req.on("end", () => {
              resolve(JSON.parse(body));
            });
            req.on("error", (err) => {
              reject(err);
            });
          }
        );

        const modelParts = body.model.split("/");

        let providerName: string;
        let model: string;
        if (modelParts.length === 1) {
          // If the user has the Anthropic provider configured,
          // proxy all requests through there instead.
          if (providers.anthropic) {
            providerName = "anthropic";
            model = modelParts[0]!;
          } else {
            // If they don't have it configured, just use
            // the normal Anthropic API.
            proxyToAnthropic(body);
          }
          return;
        } else {
          providerName = modelParts[0]!;
          model = modelParts[1]!;
        }

        const provider = providers[providerName];
        if (!provider) {
          throw new Error(`Unknown provider: ${providerName}`);
        }

        // Filter out duplicate tool calls and their corresponding results
        const filteredMessages = filterDuplicateToolCalls(body.messages);
        
        const coreMessages = convertFromAnthropicMessages(filteredMessages);
        let system: string | undefined;
        if (body.system && body.system.length > 0) {
          system = body.system.map((s) => s.text).join("\n");
        }

        const tools = body.tools?.reduce((acc, tool) => {
          acc[tool.name] = {
            description: tool.name,
            inputSchema: jsonSchema(
              providerizeSchema(providerName, tool.input_schema)
            ),
          };
          return acc;
        }, {} as Record<string, Tool>);

        const stream = streamText({
          model: provider.languageModel(model),
          system,
          tools,
          messages: coreMessages,
          maxOutputTokens: body.max_tokens,
          temperature: body.temperature,

          onFinish: ({ response, usage, finishReason }) => {
            // If the body is already being streamed,
            // we don't need to do any conversion here.
            if (body.stream) {
              return;
            }

            // There should only be one message.
            const message = response.messages[0];
            if (!message) {
              throw new Error("No message found");
            }

            const prompt = convertToAnthropicMessagesPrompt({
              prompt: [convertToLanguageModelMessage(message, {})],
              sendReasoning: true,
              warnings: [],
            });
            const promptMessage = prompt.prompt.messages[0];
            if (!promptMessage) {
              throw new Error("No prompt message found");
            }

            res.writeHead(200, { "Content-Type": "application/json" }).end(
              JSON.stringify({
                id: "msg_" + Date.now(),
                type: "message",
                role: promptMessage.role,
                content: promptMessage.content,
                model: body.model,
                stop_reason: mapAnthropicStopReason(finishReason),
                stop_sequence: null,
                usage: {
                  input_tokens: usage.inputTokens,
                  output_tokens: usage.outputTokens,
                },
              })
            );
          },
          onError: ({ error }) => {
            const statusCode = 400; // Provider errors are returned as 400
            
            // Write comprehensive debug info to temp file
            const debugFile = writeDebugToTempFile(
              statusCode,
              {
                method: "POST",
                url: req.url,
                headers: req.headers,
                body: { 
                  ...body, 
                  messages: filteredMessages,
                  _originalMessages: body.messages,
                  _originalMessageCount: body.messages.length,
                  _filteredMessageCount: filteredMessages.length
                },
              },
              {
                statusCode,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  provider: providerName,
                  model: model,
                  error: error instanceof Error ? {
                    message: error.message,
                    stack: error.stack,
                    name: error.name,
                  } : error,
                  _debugInfo: {
                    originalRequestSize: JSON.stringify(body).length,
                    filteredRequestSize: JSON.stringify({ ...body, messages: filteredMessages }).length,
                    toolCount: body.tools?.length || 0,
                    systemPromptLength: body.system?.reduce((acc, s) => acc + s.text.length, 0) || 0,
                    filteringSummary: {
                      originalMessages: body.messages.length,
                      filteredMessages: filteredMessages.length,
                      messagesRemoved: body.messages.length - filteredMessages.length
                    }
                  }
                }),
              }
            );

            if (debugFile) {
              logDebugError("Provider", statusCode, debugFile, { provider: providerName, model });
            }

            res
              .writeHead(400, {
                "Content-Type": "application/json",
              })
              .end(
                JSON.stringify({
                  type: "error",
                  error: error instanceof Error ? error.message : error,
                })
              );
          },
        });

        if (!body.stream) {
          await stream.consumeStream();
          return;
        }

        res.on("error", () => {
          // In NodeJS, this needs to be handled.
          // We already send the error to the client.
        });

        // Collect all stream chunks for debugging if enabled
        const streamChunks: any[] = [];
        const startTime = Date.now();

        await convertToAnthropicStream(stream.fullStream).pipeTo(
          new WritableStream({
            write(chunk) {
              // Collect chunks for debug dump (only in verbose mode to save memory)
              if (isVerboseDebugEnabled()) {
                streamChunks.push({
                  timestamp: Date.now() - startTime,
                  chunk: chunk
                });
              }
              
              // Check for streaming errors and log them (but don't interrupt the stream)
              if (chunk.type === "error") {
                // Always try to log errors when debug is enabled
                if (isDebugEnabled()) {
                  console.error(`[ANYCLAUDE DEBUG] Streaming error chunk detected for ${providerName}/${model} at ${Date.now() - startTime}ms:`, JSON.stringify(chunk).substring(0, 200));
                }
                
                // Write comprehensive debug info including full stream dump
                const debugFile = writeDebugToTempFile(
                  400, // Streaming errors are sent as 400
                  {
                    method: "POST",
                    url: req.url,
                    headers: req.headers,
                    body: { 
                      ...body, 
                      messages: filteredMessages,
                      _originalMessageCount: body.messages.length,
                      _filteredMessageCount: filteredMessages.length
                    },
                  },
                  {
                    statusCode: 400,
                    headers: { "Content-Type": "text/event-stream" },
                    body: JSON.stringify({
                      provider: providerName,
                      model: model,
                      streamingError: chunk,
                      fullChunk: JSON.stringify(chunk),
                      streamDuration: Date.now() - startTime,
                      streamChunkCount: streamChunks.length,
                      allStreamChunks: streamChunks,
                      _debugInfo: {
                        originalRequestSize: JSON.stringify(body).length,
                        filteredRequestSize: JSON.stringify({ ...body, messages: filteredMessages }).length,
                        toolCount: body.tools?.length || 0,
                        systemPromptLength: body.system?.reduce((acc, s) => acc + s.text.length, 0) || 0
                      }
                    }),
                  }
                );

                if (debugFile) {
                  logDebugError("Streaming", 400, debugFile, { provider: providerName, model });
                } else if (isDebugEnabled()) {
                  queueErrorMessage(`Failed to write debug file for streaming error`);
                }
              }
              
              // Write all chunks (including errors) to the stream - matching original behavior
              res.write(
                `event: ${chunk.type}\ndata: ${JSON.stringify(chunk)}\n\n`
              );
            },
            close() {
              if (isVerboseDebugEnabled() && streamChunks.length > 0) {
                console.error(`[ANYCLAUDE DEBUG] Stream completed for ${providerName}/${model}: ${streamChunks.length} chunks in ${Date.now() - startTime}ms`);
              }
              res.end();
            },
          })
        );
      })().catch((err) => {
        res.writeHead(500, {
          "Content-Type": "application/json",
        });
        res.end(
          JSON.stringify({
            error: "Internal server error: " + err.message,
          })
        );
      });
    })
    .listen(port ?? 0);

  const address = proxy.address();
  if (!address) {
    throw new Error("Failed to get proxy address");
  }
  if (typeof address === "string") {
    return address;
  }
  return `http://localhost:${address.port}`;
};
