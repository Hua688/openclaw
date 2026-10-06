import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses.js";
import { afterEach, assert, beforeEach, describe, expect, it, vi } from "vitest";
import { configureAiTransportHost, getAiTransportHost } from "../host.js";
import type { BaseOpenAIStreamOptions } from "../provider-options.js";
import { streamAzureOpenAIResponses } from "../providers/azure-openai-responses.js";
import type { Context, Model } from "../types.js";
import {
  createAzureOpenAIResponsesTransportStreamFn,
  createOpenAIResponsesTransportStreamFn,
} from "./openai-responses-client.js";

const model = {
  id: "gpt-5.6",
  name: "Cache fixture",
  api: "azure-openai-responses",
  provider: "azure",
  baseUrl: "https://fixture.openai.azure.com/openai/v1",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 32768,
  maxTokens: 256,
} satisfies Model<"azure-openai-responses">;
const png =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";
function history(): Context {
  return {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "First original." },
          { type: "image", mimeType: "image/png", data: png },
          { type: "text", text: "Last original." },
        ],
        timestamp: 1,
      },
      {
        role: "user",
        content: "Retained context.",
        runtimeContext: { retained: true },
        timestamp: 2,
      },
      {
        role: "user",
        content: "Current context.",
        runtimeContext: { retained: false },
        timestamp: 3,
      },
    ],
  };
}
function markers(body: ResponseCreateParamsStreaming) {
  if (!Array.isArray(body.input)) {
    throw new Error("Expected SDK array input");
  }
  return body.input.flatMap((item, index) =>
    "content" in item && Array.isArray(item.content)
      ? item.content.flatMap((block, blockIndex) =>
          "prompt_cache_breakpoint" in block ? [{ index, blockIndex, block }] : [],
        )
      : [],
  );
}
let previousHost: ReturnType<typeof getAiTransportHost>;
let requests: ResponseCreateParamsStreaming[];
beforeEach(() => {
  previousHost = getAiTransportHost();
  requests = [];
  configureAiTransportHost({
    buildModelFetch: () => async (input, init) => {
      requests.push((await new Request(input, init).json()) as ResponseCreateParamsStreaming);
      return Response.json({ error: { message: "synthetic capture" } }, { status: 400 });
    },
  });
});
afterEach(() => {
  configureAiTransportHost(previousHost);
  vi.unstubAllEnvs();
});
async function submit(
  entry: "managed" | "provider",
  context: Context,
  options: BaseOpenAIStreamOptions = {},
  requestModel: Model<"azure-openai-responses"> = model,
) {
  const opts = { apiKey: "synthetic-key", ...options };
  const stream =
    entry === "managed"
      ? await createAzureOpenAIResponsesTransportStreamFn()(requestModel, context, opts)
      : streamAzureOpenAIResponses(requestModel, context, opts);
  expect((await stream.result()).errorMessage).toContain("synthetic capture");
  const body = requests.at(-1);
  if (!body) {
    throw new Error("SDK did not serialize a POST");
  }
  return body;
}

describe.each(["managed", "provider"] as const)("Azure %s final SDK payload", (entry) => {
  it("binds typed instruction carriers and original user bytes through an opaque deployment and callback clone", async () => {
    vi.stubEnv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5.6=opaque-deployment");
    const context = history();
    const before = JSON.stringify(context);
    const body = await submit(entry, context, {
      onPayload: (payload) => Response.json(payload).json(),
    });
    expect(body.model).toBe("opaque-deployment");
    expect(markers(body)).toEqual([
      {
        index: 0,
        blockIndex: 2,
        block: {
          type: "input_text",
          text: "Last original.",
          prompt_cache_breakpoint: { mode: "explicit" },
        },
      },
    ]);
    expect(Array.isArray(body.input) && body.input.at(-1)).toMatchObject({
      role: "developer",
      content: [{ type: "input_text", text: "Current context." }],
    });
    expect(JSON.stringify(context)).toBe(before);
    expect(context.messages.flatMap(Object.getOwnPropertySymbols)).toEqual([]);
  });

  it.each([
    "none",
    "old-model",
    "opaque-model",
    "retained",
    "unspecified",
    "missing",
    "duplicate",
    "image-only",
  ] as const)("does not introduce a point for %s input", async (scenario) => {
    const context = history();
    let requestModel = model;
    let options: BaseOpenAIStreamOptions = {};
    if (scenario === "none") {
      options = { cacheRetention: "none" };
    } else if (scenario === "old-model" || scenario === "opaque-model") {
      requestModel = { ...model, id: scenario === "old-model" ? "gpt-5.5" : "opaque-deployment" };
    } else if (scenario === "missing") {
      context.messages.pop();
    } else if (scenario === "duplicate") {
      const current = context.messages[2];
      assert(current);
      context.messages.push({ ...current });
    } else if (scenario === "retained" || scenario === "unspecified") {
      context.messages[2] = {
        role: "user",
        content: "Current context.",
        runtimeContext: scenario === "retained" ? { retained: true } : {},
        timestamp: 3,
      };
    } else {
      context.messages[0] = {
        role: "user",
        content: [{ type: "image", mimeType: "image/png", data: png }],
        timestamp: 1,
      };
    }
    expect(markers(await submit(entry, context, options, requestModel))).toEqual([]);
  });

  it.each(["deployment", "route", "original-text", "carrier-layout"] as const)(
    "revalidates %s after the external callback",
    async (change) => {
      const body = await submit(
        entry,
        history(),
        {
          onPayload: (payload, target) => {
            const request = payload as ResponseCreateParamsStreaming;
            if (change === "deployment") {
              request.model = "other-deployment";
            } else if (change === "route") {
              target.baseUrl = "https://other.example/v1";
            } else if (Array.isArray(request.input)) {
              if (change === "original-text") {
                request.input[0] = {
                  role: "user",
                  content: [{ type: "input_text", text: "Synthesized user." }],
                };
              } else {
                request.input.reverse();
              }
            }
          },
        },
        { ...model },
      );
      expect(markers(body)).toEqual([]);
    },
  );

  it("preserves caller-supplied points without adding another", async () => {
    const body = await submit(entry, history(), {
      onPayload: (payload) => {
        const request = payload as ResponseCreateParamsStreaming;
        const item = Array.isArray(request.input) && request.input[0];
        const block = item && "content" in item && Array.isArray(item.content) && item.content[0];
        if (block && block.type === "input_text") {
          block.prompt_cache_breakpoint = { mode: "explicit" };
        }
      },
    });
    expect(markers(body)).toEqual([
      {
        index: 0,
        blockIndex: 0,
        block: {
          type: "input_text",
          text: "First original.",
          prompt_cache_breakpoint: { mode: "explicit" },
        },
      },
    ]);
  });
});

it("ignores image-cleanup placeholder text at the managed SDK boundary", async () => {
  const context = history();
  context.messages[0] = {
    role: "user",
    content: [
      { type: "text", text: "Original beside invalid image." },
      { type: "image", mimeType: "image/png", data: "SGVsbG8=" },
    ],
    timestamp: 1,
  };
  const body = await submit("managed", context);
  expect(markers(body)).toEqual([
    {
      index: 0,
      blockIndex: 0,
      block: {
        type: "input_text",
        text: "Original beside invalid image.",
        prompt_cache_breakpoint: { mode: "explicit" },
      },
    },
  ]);
  expect(Array.isArray(body.input) && body.input[0]).toMatchObject({
    content: [expect.anything(), { type: "input_text", text: expect.any(String) }],
  });
});

it("keeps direct OpenAI SDK requests marker-free", async () => {
  const stream = await createOpenAIResponsesTransportStreamFn()(
    {
      ...model,
      api: "openai-responses",
      provider: "openai",
      baseUrl: "https://api.openai.com/v1",
    },
    history(),
    { apiKey: "synthetic-key" },
  );
  expect((await stream.result()).errorMessage).toContain("synthetic capture");
  assert(requests[0]);
  expect(markers(requests[0])).toEqual([]);
});
