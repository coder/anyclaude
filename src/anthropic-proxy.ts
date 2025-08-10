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
  queueErrorMessage,
  debug
} from "./debug";

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

        const coreMessages = convertFromAnthropicMessages(body.messages);
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
            let statusCode = 400; // Provider errors are returned as 400
            let transformedError = error;
            
            // Check if this is an OpenAI server error that we should transform
            const isOpenAIServerError = providerName === 'openai' && 
              error && typeof error === 'object' && 
              'error' in error && (error as any).error?.code === 'server_error';
            
            if (isOpenAIServerError) {
              debug(1, `OpenAI server error detected in onError for ${model}. Transforming to 429 to trigger retry...`);
              // Transform to rate limit error to trigger retry
              statusCode = 429;
              transformedError = {
                type: "error",
                error: {
                  type: "rate_limit_error", 
                  message: "OpenAI server temporarily unavailable. Please retry your request."
                }
              };
            }
            
            // Write comprehensive debug info to temp file
            const debugFile = writeDebugToTempFile(
              statusCode,
              {
                method: "POST",
                url: req.url,
                headers: req.headers,
                body: body,
              },
              {
                statusCode,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  provider: providerName,
                  model: model,
                  originalError: error instanceof Error ? {
                    message: error.message,
                    stack: error.stack,
                    name: error.name,
                  } : error,
                  error: transformedError instanceof Error ? {
                    message: transformedError.message,
                    stack: transformedError.stack,
                    name: transformedError.name,
                  } : transformedError,
                  wasTransformed: isOpenAIServerError,
                  _debugInfo: {
                    requestSize: JSON.stringify(body).length,
                    toolCount: body.tools?.length || 0,
                    systemPromptLength: body.system?.reduce((acc, s) => acc + s.text.length, 0) || 0,
                    messageCount: body.messages.length
                  }
                }),
              }
            );

            if (debugFile) {
              logDebugError("Provider", statusCode, debugFile, { provider: providerName, model });
            }

            res
              .writeHead(statusCode, {
                "Content-Type": "application/json",
              })
              .end(
                JSON.stringify({
                  type: "error",
                  error: transformedError instanceof Error ? transformedError.message : transformedError,
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
                // Store original error for debugging
                const originalError = { ...chunk };
                
                // Check if this is an OpenAI server error (any sequence)
                const isOpenAIServerError = providerName === 'openai' && 
                  (chunk as any).error?.code === 'server_error';
                
                if (isOpenAIServerError) {
                  debug(1, `OpenAI server error detected for ${model} at sequence ${(chunk as any).sequence_number}. This is a known transient issue with OpenAI.`);
                  debug(1, `Transforming to 429 rate limit error to trigger Claude Code's automatic retry...`);
                  
                  // Transform OpenAI server errors to 429 rate limit errors
                  // This should trigger Claude Code's built-in retry mechanism
                  chunk = {
                    type: "error",
                    sequence_number: (chunk as any).sequence_number,
                    error: {
                      type: "rate_limit_error" as any,
                      code: "rate_limit_error",
                      message: "OpenAI server temporarily unavailable. Please retry your request.",
                      param: null
                    }
                  } as any;
                } else {
                  // Log other errors normally
                  debug(1, `Streaming error chunk detected for ${providerName}/${model} at ${Date.now() - startTime}ms:`, chunk);
                }
                
                // Write comprehensive debug info including full stream dump
                const debugFile = writeDebugToTempFile(
                  400, // Streaming errors are sent as 400
                  {
                    method: "POST",
                    url: req.url,
                    headers: req.headers,
                    body: body,
                  },
                  {
                    statusCode: 400,
                    headers: { "Content-Type": "text/event-stream" },
                    body: JSON.stringify({
                      provider: providerName,
                      model: model,
                      streamingError: originalError,
                      transformedError: isOpenAIServerError ? chunk : null,
                      wasTransformed: isOpenAIServerError,
                      fullChunk: JSON.stringify(originalError),
                      streamDuration: Date.now() - startTime,
                      streamChunkCount: streamChunks.length,
                      allStreamChunks: streamChunks,
                      _debugInfo: {
                        requestSize: JSON.stringify(body).length,
                        toolCount: body.tools?.length || 0,
                        systemPromptLength: body.system?.reduce((acc, s) => acc + s.text.length, 0) || 0,
                        messageCount: body.messages.length
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
              if (streamChunks.length > 0) {
                debug(2, `Stream completed for ${providerName}/${model}: ${streamChunks.length} chunks in ${Date.now() - startTime}ms`);
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
