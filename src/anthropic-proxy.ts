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

        const proxy = https.request(
          {
            host: "api.anthropic.com",
            path: req.url,
            method: req.method,
            headers: req.headers,
          },
          (proxiedRes) => {
            res.writeHead(proxiedRes.statusCode ?? 500, proxiedRes.headers);
            proxiedRes.pipe(res, {
              end: true,
            });
          }
        );
        if (body) {
          proxy.end(JSON.stringify(body));
        } else {
          req.pipe(proxy, {
            end: true,
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

        await convertToAnthropicStream(stream.fullStream).pipeTo(
          new WritableStream({
            write(chunk) {
              res.write(
                `event: ${chunk.type}\ndata: ${JSON.stringify(chunk)}\n\n`
              );
            },
            close() {
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
