import path from "node:path";
import { configureAiTransportHost, getAiTransportHost } from "@openclaw/ai";
import {
  createAzureOpenAIResponsesTransportStreamFn,
  createOpenAIResponsesTransportStreamFn,
} from "@openclaw/ai/transports";
import type { Context, Model } from "@openclaw/llm-core";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import { assert, expect, it } from "vitest";
import { convertToLlm } from "../../../packages/agent-core/src/harness/messages.js";
import {
  buildOpenAIResponsesReasoningReplayMetadata,
  captureOpenAIResponsesCompaction,
} from "../../../packages/ai/src/transports/openai-responses-compaction-replay.js";
import { createOpenAIResponsesAssistantOutput } from "../../../packages/ai/src/transports/openai-responses-replay-messages-internal.js";
import { upsertSessionEntryCore } from "../../config/sessions/session-accessor.js";
import {
  withOpenClawTestState,
  type OpenClawTestState,
} from "../../test-utils/openclaw-test-state.js";
import { SessionManager } from "./session-manager.js";

const tool = {
  name: "lookup",
  description: "Look up a value.",
  parameters: { type: "object", properties: { value: { type: "string" } } },
};
const reasoning = {
  type: "reasoning",
  id: "rs_cache_stability",
  status: "completed",
  summary: [{ type: "summary_text", text: "Checking the requested value." }],
  content: [],
};
const ciphertext = "synthetic_reasoning_ciphertext";
const call = {
  type: "function_call",
  id: "fc_cache_stability",
  call_id: "call_cache_stability",
  name: tool.name,
  arguments: '{"value":"original"}',
  status: "completed",
};

function createHarness(api: "openai-responses" | "azure-openai-responses") {
  const previousHost = getAiTransportHost();
  const requests: Record<string, unknown>[][] = [];
  const thinkingEvents: string[] = [];
  const model: Model = {
    id: "cache-stability-model",
    name: "Cache stability fixture",
    api,
    provider: api === "openai-responses" ? "openai" : "azure",
    baseUrl:
      api === "openai-responses"
        ? "https://api.openai.com/v1"
        : "https://fixture.openai.azure.com/openai/v1",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32_768,
    maxTokens: 256,
  };
  configureAiTransportHost({
    buildModelFetch: () => async (input, init) => {
      const request = new Request(input, init);
      const body: unknown = await request.json();
      assert(isRecord(body) && Array.isArray(body.input));
      requests.push(body.input.filter(isRecord));
      const events = [
        { type: "response.output_item.done", output_index: 0, item: reasoning },
        { type: "response.output_item.done", output_index: 1, item: call },
        {
          type: "response.completed",
          response: {
            id: `resp_cache_${requests.length}`,
            status: "completed",
            output: [{ ...reasoning, encrypted_content: ciphertext }, call],
            usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          },
        },
      ];
      return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
        headers: { "content-type": "text/event-stream" },
      });
    },
  });
  return {
    requests,
    thinkingEvents,
    model,
    run: async (messages: Context["messages"]) => {
      const factory =
        api === "azure-openai-responses"
          ? createAzureOpenAIResponsesTransportStreamFn
          : createOpenAIResponsesTransportStreamFn;
      const stream = await factory()(
        model,
        { systemPrompt: "Keep the request history stable.", messages, tools: [tool] },
        { apiKey: "synthetic-key", sessionId: "cache-stability", transport: "sse" },
      );
      for await (const event of stream) {
        if (event.type === "thinking_end") {
          thinkingEvents.push(event.content);
        }
      }
      return stream.result();
    },
    close: () => configureAiTransportHost(previousHost),
  };
}

async function openTestSession(state: OpenClawTestState) {
  const target = {
    agentId: "main",
    sessionId: "cache-stability",
    sessionKey: "agent:main:cache-stability",
    storePath: path.join(state.agentDir("main"), "openclaw-agent.sqlite"),
  };
  await upsertSessionEntryCore(target, { sessionId: target.sessionId, updatedAt: 1 });
  return {
    manager: SessionManager.open(target, state.workspaceDir),
    reopen: () => SessionManager.open(target, state.workspaceDir),
  };
}

it.each(["openai-responses", "azure-openai-responses"] as const)(
  "%s keeps streamed reasoning replay identical after SQLite reload without losing display text",
  async (api) => {
    const harness = createHarness(api);
    try {
      await withOpenClawTestState({ label: "responses-cache-stability" }, async (state) => {
        const { manager, reopen } = await openTestSession(state);
        const user = { role: "user" as const, content: "Look up the value.", timestamp: 1 };
        manager.appendMessage(user);
        const assistant = await harness.run([user]);
        expect(assistant.stopReason).toBe("toolUse");
        const thinking = assistant.content.find((block) => block.type === "thinking");
        assert(thinking);
        expect(thinking.thinking).toBe("Checking the requested value.");
        expect(harness.thinkingEvents).toEqual(["Checking the requested value."]);
        assert(thinking.thinkingSignature);
        expect(JSON.parse(thinking.thinkingSignature)).toEqual({
          id: reasoning.id,
          type: "reasoning",
          status: "completed",
          summary: [],
          encrypted_content: ciphertext,
        });
        const toolCall = assistant.content.find((block) => block.type === "toolCall");
        assert(toolCall);
        const result = {
          role: "toolResult" as const,
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          content: [{ type: "text" as const, text: "found" }],
          isError: false,
          timestamp: 2,
        };
        manager.appendMessage(assistant);
        manager.appendMessage(result);
        await harness.run([user, assistant, result]);
        const reopened = reopen();
        await harness.run(convertToLlm(reopened.buildSessionContext().messages));
        const live = harness.requests[1];
        const reloaded = harness.requests[2];
        assert(live && reloaded);
        expect(reloaded).toEqual(live);
        expect(live.find((item) => item.type === "reasoning")).toMatchObject({
          summary: [],
          encrypted_content: ciphertext,
        });
        expect(live.find((item) => item.type === "reasoning")).not.toHaveProperty("content");
        expect(toolCall.arguments).toEqual({ value: "original" });
      });
    } finally {
      harness.close();
    }
  },
);

it.each(["openai-responses", "azure-openai-responses"] as const)(
  "%s sends current context last through compacted tools and a SQLite-reloaded new turn",
  async (api) => {
    const harness = createHarness(api);
    const user = { role: "user" as const, content: "Look up the value.", timestamp: 1 };
    const carrier = {
      role: "user" as const,
      content: "current runtime snapshot",
      timestamp: 1,
      runtimeContextCarrier: true,
      runtimeContextCarrierRetained: false,
    };
    try {
      await withOpenClawTestState({ label: "responses-carrier-stability" }, async (state) => {
        const { manager, reopen } = await openTestSession(state);
        const checkpoint = createOpenAIResponsesAssistantOutput(harness.model);
        const compacted = {
          type: "compaction" as const,
          id: "cmp_cache_stability",
          encrypted_content: "synthetic_compaction_ciphertext",
        };
        captureOpenAIResponsesCompaction(
          checkpoint,
          compacted,
          "retained-users",
          harness.model,
          buildOpenAIResponsesReasoningReplayMetadata(harness.model, {
            sessionId: "cache-stability",
          }),
          [
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: user.content }],
            },
            compacted,
          ],
        );
        const history = [user, checkpoint];
        manager.appendMessage(user);
        manager.appendMessage(checkpoint);
        const assistant = await harness.run([...history, carrier]);
        const toolCall = assistant.content.find((block) => block.type === "toolCall");
        assert(toolCall);
        const result = {
          role: "toolResult" as const,
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          content: [{ type: "text" as const, text: "found" }],
          isError: false,
          timestamp: 2,
        };
        manager.appendMessage(assistant);
        manager.appendMessage(result);
        await harness.run([...history, carrier, assistant, result]);
        const nextCarrier = { ...carrier, content: "updated runtime snapshot", timestamp: 3 };
        const reopened = reopen();
        reopened.appendMessage({ role: "user", content: "Continue.", timestamp: 3 });
        await harness.run([...convertToLlm(reopened.buildSessionContext().messages), nextCarrier]);
        const [first, followup, next] = harness.requests;
        assert(first && followup && next);
        expect(followup.at(-1)).toEqual(first.at(-1));
        expect(followup.map((item) => item.type)).toEqual([
          ...(api === "azure-openai-responses" ? ["message"] : []),
          "message",
          "compaction",
          "reasoning",
          "function_call",
          "function_call_output",
          "message",
        ]);
        expect(next.slice(0, followup.length - 1)).toEqual(followup.slice(0, -1));
        expect(next.at(-1)).toMatchObject({
          role: "user",
          content: [{ type: "input_text", text: nextCarrier.content }],
        });
        expect(next).not.toContainEqual(first.at(-1));
        expect(
          followup.filter((item) => JSON.stringify(item).includes(carrier.content)),
        ).toHaveLength(1);
        expect(followup.find((item) => item.type === "compaction")).toEqual(compacted);
      });
    } finally {
      harness.close();
    }
  },
);
