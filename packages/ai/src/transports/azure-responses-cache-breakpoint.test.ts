import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { configureAiTransportHost, getAiTransportHost } from "../host.js";
import type { BaseOpenAIStreamOptions } from "../provider-options.js";
import { streamAzureOpenAIResponses } from "../providers/azure-openai-responses.js";
import type { Context, Model } from "../types.js";
import { createZeroUsage } from "../usage.test-support.js";
import {
  createAzureOpenAIResponsesTransportStreamFn,
  createOpenAIResponsesTransportStreamFn,
} from "./openai-responses-client.js";
import { buildOpenAIResponsesReasoningReplayMetadata } from "./openai-responses-compaction-replay.js";
import {
  buildOpenAIResponsesReasoningSignature,
  type OpenAIResponsesCompactionReplayState,
} from "./openai-responses-contracts.js";

const model = {
  id: "gpt-5.6",
  name: "GPT-5.6",
  api: "azure-openai-responses",
  provider: "azure",
  baseUrl: "https://example.openai.azure.com/openai/v1",
  reasoning: false,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 200000,
  maxTokens: 8192,
} satisfies Model<"azure-openai-responses">;

const png =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

function history(): Context {
  return {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Earlier reusable text." },
          { type: "image", mimeType: "image/png", data: png },
          { type: "text", text: "Last reusable text." },
        ],
        timestamp: 1,
      },
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: "Display reasoning.",
            thinkingSignature: buildOpenAIResponsesReasoningSignature(
              { id: "rs_fixture", encrypted_content: "opaque-fixture" },
              buildOpenAIResponsesReasoningReplayMetadata(model),
            ),
          },
          {
            type: "toolCall",
            id: "call_fixture|fc_fixture",
            name: "inspect_fixture",
            arguments: { section: "synthetic" },
          },
        ],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: createZeroUsage(),
        stopReason: "toolUse",
        timestamp: 2,
      },
      {
        role: "toolResult",
        toolCallId: "call_fixture|fc_fixture",
        toolName: "inspect_fixture",
        content: [{ type: "text", text: "Synthetic result." }],
        isError: false,
        timestamp: 3,
      },
      {
        role: "user",
        content: "Retained facts.",
        runtimeContextCarrier: true,
        runtimeContextCarrierRetained: true,
        timestamp: 4,
      },
      {
        role: "user",
        content: "Current facts.",
        runtimeContextCarrier: true,
        timestamp: 5,
      },
    ],
  };
}

function markers(body: ResponseCreateParamsStreaming) {
  const result: Array<{ item: number; block: number; text: string | undefined }> = [];
  if (!Array.isArray(body.input)) {
    throw new Error("Expected actual SDK array input");
  }
  body.input.forEach((item, itemIndex) => {
    if (!("content" in item) || !Array.isArray(item.content)) {
      return;
    }
    item.content.forEach((block, blockIndex) => {
      if ("prompt_cache_breakpoint" in block) {
        expect(block.prompt_cache_breakpoint).toEqual({ mode: "explicit" });
        result.push({
          item: itemIndex,
          block: blockIndex,
          text: block.type === "input_text" ? block.text : undefined,
        });
      }
    });
  });
  return result;
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
  const streamOptions = { apiKey: "synthetic-test-key", ...options };
  const stream =
    entry === "managed"
      ? await createAzureOpenAIResponsesTransportStreamFn()(requestModel, context, streamOptions)
      : streamAzureOpenAIResponses(requestModel, context, streamOptions);
  const result = await stream.result();
  expect(result.errorMessage).toContain("synthetic capture");
  const request = requests.at(-1);
  if (!request) {
    throw new Error("SDK did not serialize a POST");
  }
  return request;
}

describe.each(["managed", "provider"] as const)("Azure %s SDK cache breakpoint", (entry) => {
  it("marks the last original text before transient context without rewriting replay", async () => {
    const context = history();
    const sourceBytes = JSON.stringify(context);
    const baseline = await submit(entry, context, { cacheRetention: "none" });
    const marked = await submit(entry, context, { cacheRetention: "short" });
    expect(requests).toHaveLength(2);
    expect(markers(baseline)).toEqual([]);
    expect(markers(marked)).toEqual([{ item: 0, block: 2, text: "Last reusable text." }]);
    const projected = structuredClone(marked);
    if (!Array.isArray(projected.input)) {
      throw new Error("Expected serialized input");
    }
    for (const item of projected.input) {
      if ("content" in item && Array.isArray(item.content)) {
        for (const block of item.content) {
          if ("prompt_cache_breakpoint" in block) {
            delete block.prompt_cache_breakpoint;
          }
        }
      }
    }
    expect(projected).toEqual(baseline);
    expect(projected.input.at(-1)).toMatchObject({
      role: "user",
      content: [{ type: "input_text", text: "Current facts." }],
    });
    expect(projected.input).toContainEqual(
      expect.objectContaining({
        type: "reasoning",
        summary: [],
        encrypted_content: "opaque-fixture",
      }),
    );
    expect(JSON.stringify(context)).toBe(sourceBytes);
    for (const message of context.messages) {
      expect(Object.getOwnPropertySymbols(message)).toEqual([]);
    }
  });

  it("uses the logical model family with an opaque mapped deployment and transparent callback", async () => {
    vi.stubEnv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5.6=opaque-deployment");
    const body = await submit(entry, history(), {
      onPayload: (payload) => Response.json(payload).json(),
    });
    expect(body.model).toBe("opaque-deployment");
    expect(markers(body)).toEqual([{ item: 0, block: 2, text: "Last reusable text." }]);
  });

  it.each(["gpt-5.5", "opaque-deployment"])(
    "leaves unsupported logical model %s unchanged",
    async (id) => {
      const requestModel = { ...model, id };
      const baseline = await submit(entry, history(), { cacheRetention: "none" }, requestModel);
      const body = await submit(entry, history(), {}, requestModel);
      expect(body).toEqual(baseline);
      expect(markers(body)).toEqual([]);
    },
  );

  it("does not regrant a breakpoint after the callback changes the deployment", async () => {
    const body = await submit(entry, history(), {
      onPayload: (payload) => ({
        ...(payload as ResponseCreateParamsStreaming),
        model: "different-deployment",
      }),
    });
    expect(body.model).toBe("different-deployment");
    expect(markers(body)).toEqual([]);
  });

  it("does not regrant a breakpoint after the callback changes the prepared route", async () => {
    const requestModel = { ...model };
    const body = await submit(
      entry,
      history(),
      {
        onPayload: (payload, target) => {
          target.baseUrl = "https://different.example/openai/v1";
          return payload;
        },
      },
      requestModel,
    );
    expect(markers(body)).toEqual([]);
  });

  it("preserves caller-supplied markers without adding another", async () => {
    const body = await submit(entry, history(), {
      onPayload: (payload) => {
        const request = payload as ResponseCreateParamsStreaming;
        if (Array.isArray(request.input)) {
          const first = request.input[0];
          if (first && "content" in first && Array.isArray(first.content)) {
            const block = first.content[0];
            if (block?.type === "input_text") {
              block.prompt_cache_breakpoint = { mode: "explicit" };
            }
          }
        }
      },
    });
    expect(markers(body)).toEqual([{ item: 0, block: 0, text: "Earlier reusable text." }]);
  });

  it("skips image-only original users rather than marking carrier text", async () => {
    const context = history();
    context.messages[0] = {
      role: "user",
      content: [{ type: "image", mimeType: "image/png", data: png }],
      timestamp: 1,
    };
    const body = await submit(entry, context);
    expect(markers(body)).toEqual([]);
  });

  it("skips callback-created user text rather than treating it as original history", async () => {
    const body = await submit(entry, history(), {
      onPayload: (payload) => {
        const request = payload as ResponseCreateParamsStreaming;
        if (Array.isArray(request.input)) {
          request.input[0] = {
            role: "user",
            content: [{ type: "input_text", text: "Replacement." }],
          };
        }
      },
    });
    expect(markers(body)).toEqual([]);
  });

  it.each(["missing", "retained", "duplicate"] as const)(
    "preserves the %s current-carrier layout",
    async (layout) => {
      const context = history();
      const current = context.messages.at(-1);
      if (!current || current.role !== "user") {
        throw new Error("Expected current carrier");
      }
      if (layout === "missing") {
        context.messages.pop();
      } else if (layout === "retained") {
        current.runtimeContextCarrierRetained = true;
      } else {
        context.messages.push({ ...current });
      }
      const baseline = await submit(entry, context, { cacheRetention: "none" });
      const body = await submit(entry, context);
      expect(body).toEqual(baseline);
      expect(markers(body)).toEqual([]);
    },
  );

  it("preserves canonical compaction replay without marking saved user items", async () => {
    const context = history();
    const assistant = context.messages[1];
    if (!assistant || assistant.role !== "assistant") {
      throw new Error("Expected assistant");
    }
    const metadata = buildOpenAIResponsesReasoningReplayMetadata(model);
    if (!metadata.baseUrlHash) {
      throw new Error("Expected replayable fixture URL");
    }
    const replay: OpenAIResponsesCompactionReplayState = {
      ...metadata,
      baseUrlHash: metadata.baseUrlHash,
      v: 1,
      type: "openai-responses-retained-compaction",
      id: "cmp_fixture",
      data: "opaque-compaction",
      compactedWindow: {
        state: "ready",
        output: JSON.stringify([
          {
            role: "user",
            type: "message",
            content: [{ type: "input_text", text: "Canonical user." }],
          },
          { type: "compaction", id: "cmp_fixture", encrypted_content: "opaque-compaction" },
        ]),
      },
    };
    assistant.providerReplay = replay;
    const baseline = await submit(entry, context, { cacheRetention: "none" });
    const body = await submit(entry, context);
    expect(body).toEqual(baseline);
    expect(body.input).toContainEqual({
      role: "user",
      type: "message",
      content: [{ type: "input_text", text: "Canonical user." }],
    });
    expect(markers(body)).toEqual([]);
  });

  it("advances to a new original user turn rather than freezing the previous read-only point", async () => {
    const context = history();
    const first = await submit(entry, context);
    context.messages.splice(-1, 0, {
      role: "user",
      content: "Next original user turn.",
      timestamp: 6,
    });
    const next = await submit(entry, context);
    expect(markers(first)).toEqual([{ item: 0, block: 2, text: "Last reusable text." }]);
    expect(markers(next)).toEqual([{ item: 5, block: 0, text: "Next original user turn." }]);
  });

  it("skips an altered carrier layout after the external payload callback", async () => {
    const body = await submit(entry, history(), {
      onPayload: (payload) => {
        const request = payload as ResponseCreateParamsStreaming;
        if (!Array.isArray(request.input)) {
          throw new Error("Expected converted input");
        }
        const current = request.input.pop();
        if (!current) {
          throw new Error("Expected current carrier");
        }
        request.input.unshift(current);
      },
    });
    expect(markers(body)).toEqual([]);
  });
});

it("does not mark image-sanitizer replacement text at the managed SDK boundary", async () => {
  const context = history();
  context.messages[0] = {
    role: "user",
    content: [
      { type: "text", text: "Real text." },
      { type: "image", mimeType: "image/png", data: "SGVsbG8=" },
    ],
    timestamp: 1,
  };
  const body = await submit("managed", context);
  expect(markers(body)).toEqual([{ item: 0, block: 0, text: "Real text." }]);
  expect(body.input).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        content: [
          { type: "input_text", text: "Real text.", prompt_cache_breakpoint: { mode: "explicit" } },
          expect.objectContaining({ type: "input_text" }),
        ],
      }),
    ]),
  );
});

it("leaves the direct OpenAI Responses SDK request unchanged", async () => {
  const openaiModel = {
    ...model,
    api: "openai-responses",
    provider: "openai",
    baseUrl: "https://api.openai.com/v1",
  } satisfies Model<"openai-responses">;
  const stream = await createOpenAIResponsesTransportStreamFn()(openaiModel, history(), {
    apiKey: "synthetic-test-key",
  });
  expect((await stream.result()).errorMessage).toContain("synthetic capture");
  expect(requests).toHaveLength(1);
  const body = requests[0];
  if (!body) {
    throw new Error("Expected SDK request");
  }
  expect(markers(body)).toEqual([]);
});
