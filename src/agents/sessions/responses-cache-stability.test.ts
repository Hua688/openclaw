import path from "node:path";
import { Type } from "typebox";
import { afterEach, assert, expect, it } from "vitest";
import { configureAiTransportHost, getAiTransportHost } from "../../../packages/ai/src/host.js";
import { streamAzureOpenAIResponses } from "../../../packages/ai/src/providers/azure-openai-responses.js";
import { createAzureOpenAIResponsesTransportStreamFn } from "../../../packages/ai/src/transports/openai-responses-client.js";
import type { Context, Model } from "../../../packages/ai/src/types.js";
import {
  isRuntimeContextMessage,
  setRuntimeContextRetention,
} from "../../../packages/llm-core/src/types.js";
import { upsertSessionEntryCore } from "../../config/sessions/session-accessor.js";
import { withOpenClawTestState } from "../../test-utils/openclaw-test-state.js";
import {
  installRuntimeContextMessageForPrompt,
  normalizeMessagesForLlmBoundary,
} from "../embedded-agent-runner/run/attempt-llm-boundary.js";
import { buildRuntimeContextCustomMessage } from "../embedded-agent-runner/run/runtime-context-prompt.js";
import type { AgentMessage } from "../runtime/index.js";
import {
  createTestSession,
  registerAgentSessionLoopTestLifecycle,
  streamMocks,
} from "./agent-session-loop-correctness.test-support.js";
import { convertToLlm } from "./messages.js";
import { SessionManager } from "./session-manager.js";

registerAgentSessionLoopTestLifecycle();

const model = {
  id: "gpt-5.6-luna",
  name: "Cache stability fixture",
  api: "azure-openai-responses",
  provider: "azure",
  baseUrl: "https://fixture.openai.azure.com/openai/v1",
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 32768,
  maxTokens: 256,
} satisfies Model<"azure-openai-responses">;
const previousHost = getAiTransportHost();
afterEach(() => configureAiTransportHost(previousHost));

type WireItem = { type: string; role?: string; content?: Array<Record<string, unknown>> };

it.each(["managed", "provider"] as const)(
  "%s keeps one typed current context and canonical reasoning through five SQLite-reloaded tool turns",
  async (entry) => {
    const requests: WireItem[][] = [];
    const display: string[] = [];
    configureAiTransportHost({
      buildModelFetch: () => async (input, init) => {
        const body = (await new Request(input, init).json()) as { input: WireItem[] };
        requests.push(body.input);
        const index = requests.length;
        const toolTurn = index % 2 === 1;
        const reasoning = {
          type: "reasoning",
          id: `rs_${index}`,
          status: "completed",
          summary: [{ type: "summary_text", text: "Synthetic display reasoning." }],
          content: [],
          ...(toolTurn ? { encrypted_content: `opaque_${index}` } : {}),
        };
        const call = {
          type: "function_call",
          id: `fc_${index}`,
          call_id: `call_${index}`,
          name: "lookup",
          arguments: '{"value":"original"}',
          status: "completed",
        };
        const text = {
          type: "message",
          id: `msg_${index}`,
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "Synthetic answer.", annotations: [] }],
        };
        const terminalReasoning = { ...reasoning, encrypted_content: `opaque_${index}` };
        const terminal = toolTurn ? call : text;
        const events = [
          { type: "response.output_item.done", output_index: 0, item: reasoning },
          { type: "response.output_item.done", output_index: 1, item: terminal },
          {
            type: "response.completed",
            response: {
              id: `resp_${index}`,
              model: model.id,
              status: "completed",
              output: [terminalReasoning, terminal],
              usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
            },
          },
        ];
        return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
          headers: { "content-type": "text/event-stream" },
        });
      },
    });
    const run = async (messages: AgentMessage[]) => {
      const context: Context = {
        systemPrompt: "Stable fixture instructions.",
        messages: convertToLlm(
          normalizeMessagesForLlmBoundary(messages, {
            sessionVersion: 4,
            includeTimestamp: false,
          }),
        ),
        tools: [
          {
            name: "lookup",
            description: "Lookup fixture.",
            parameters: { type: "object", properties: { value: { type: "string" } } },
          },
        ],
      };
      expect(context.messages.filter((message) => "runtimeContext" in message)).toHaveLength(1);
      for (const message of context.messages) {
        if (isRuntimeContextMessage(message)) {
          setRuntimeContextRetention(message, false);
        }
      }
      const options = {
        apiKey: "synthetic-key",
        sessionId: "cache-stability",
        transport: "sse" as const,
      };
      const stream =
        entry === "managed"
          ? await createAzureOpenAIResponsesTransportStreamFn()(model, context, options)
          : streamAzureOpenAIResponses(model, context, options);
      for await (const event of stream) {
        if (event.type === "thinking_end") {
          display.push(event.content);
        }
      }
      return stream.result();
    };
    await withOpenClawTestState({ label: "responses-cache-beta" }, async (state) => {
      const target = {
        agentId: "main",
        sessionId: "cache-stability",
        sessionKey: "agent:main:cache-stability",
        storePath: path.join(state.agentDir("main"), "openclaw-agent.sqlite"),
      };
      await upsertSessionEntryCore(target, { sessionId: target.sessionId, updatedAt: 1 });
      let previous: WireItem[] = [];
      for (let turn = 0; turn < 5; turn++) {
        const manager = await SessionManager.openAsync(target, state.workspaceDir);
        const user = {
          role: "user" as const,
          content: `Question ${turn}.`,
          timestamp: turn * 10 + 1,
          idempotencyKey: `fixture_${turn}`,
        };
        await manager.appendMessageAsync(user);
        let messages = manager.buildSessionContext().messages;
        const carrier = buildRuntimeContextCustomMessage(`Current facts ${turn}.`);
        assert(carrier);
        const agent = { state: { messages } };
        const cleanup = installRuntimeContextMessageForPrompt({
          session: {
            get messages() {
              return agent.state.messages;
            },
            agent,
          },
          message: carrier,
          persistedUserIdempotencyKey: user.idempotencyKey,
        });
        messages = agent.state.messages;
        try {
          const assistant = await run(messages);
          expect(assistant.stopReason).toBe("toolUse");
          const first = requests.at(-1);
          assert(first);
          expect(first.at(-1)).toMatchObject({
            role: "developer",
            content: [
              {
                type: "input_text",
                text: expect.stringContaining(`Current facts ${turn}.`),
              },
            ],
          });
          expect(
            first.filter((item) => JSON.stringify(item).includes("Current facts")),
          ).toHaveLength(1);
          const marked = first.flatMap(
            (item) => item.content?.filter((block) => block.prompt_cache_breakpoint) ?? [],
          );
          expect(marked).toEqual([
            {
              type: "input_text",
              text: user.content,
              prompt_cache_breakpoint: { mode: "explicit" },
            },
          ]);
          expect(first.slice(0, previous.length)).toEqual(previous);
          const thinking = assistant.content.find((block) => block.type === "thinking");
          assert(thinking?.thinkingSignature);
          expect(JSON.parse(thinking.thinkingSignature)).toEqual({
            id: `rs_${requests.length}`,
            type: "reasoning",
            status: "completed",
            summary: [],
            encrypted_content: `opaque_${requests.length}`,
          });
          expect(thinking).toHaveProperty("openclawReasoningReplay");
          const call = assistant.content.find((block) => block.type === "toolCall");
          assert(call);
          expect(call.arguments).toEqual({ value: "original" });
          const result = {
            role: "toolResult" as const,
            toolCallId: call.id,
            toolName: call.name,
            content: [{ type: "text" as const, text: "found" }],
            isError: false,
            timestamp: turn * 10 + 2,
          };
          await manager.appendMessageAsync(assistant);
          await manager.appendMessageAsync(result);
          messages.push(assistant, result);
          const final = await run(messages);
          expect(final.stopReason).toBe("stop");
          const followup = requests.at(-1);
          assert(followup);
          expect(followup.at(-1)).toEqual(first.at(-1));
          expect(followup.slice(0, first.length - 1)).toEqual(first.slice(0, -1));
          await manager.appendMessageAsync(final);
          const reopened = await SessionManager.openAsync(target, state.workspaceDir);
          const reloaded = reopened.buildSessionContext().messages;
          const liveReplay = convertToLlm(
            normalizeMessagesForLlmBoundary([...messages, final], {
              sessionVersion: 4,
              includeTimestamp: false,
            }),
          ).filter((message) => !("runtimeContext" in message));
          expect(
            convertToLlm(
              normalizeMessagesForLlmBoundary(reloaded, {
                sessionVersion: 4,
                includeTimestamp: false,
              }),
            ),
          ).toEqual(liveReplay);
          // Next-turn comparison excludes the retired marker: the newest original user owns the point.
          previous = followup.slice(0, -1).map((item) =>
            Object.assign(
              {},
              item,
              item.content
                ? {
                    content: item.content.map(
                      ({ prompt_cache_breakpoint: _point, ...block }) => block,
                    ),
                  }
                : {},
            ),
          );
        } finally {
          cleanup();
        }
        expect(agent.state.messages).not.toContain(carrier);
      }
    });
    expect(requests).toHaveLength(10);
    expect(display).toEqual(Array.from({ length: 10 }, () => "Synthetic display reasoning."));
  },
);

it.each(["managed", "provider"] as const)(
  "%s keeps ordered transient context at the tail through tool-busy and model-busy steering",
  async (entry) => {
    const requests: WireItem[][] = [];
    let session: Awaited<ReturnType<typeof createTestSession>>["session"];
    configureAiTransportHost({
      buildModelFetch: () => async (input, init) => {
        const body = (await new Request(input, init).json()) as { input: WireItem[] };
        requests.push(body.input);
        const index = requests.length;
        if (index === 2) {
          await session.steer(
            "Second steering question.",
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            { text: "Fresh permission and conversation facts." },
          );
        }
        const reasoning = {
          type: "reasoning",
          id: `rs_steer_${index}`,
          status: "completed",
          summary: [],
          encrypted_content: `opaque_steer_${index}`,
        };
        const output =
          index < 4
            ? {
                type: "function_call",
                id: `fc_steer_${index}`,
                call_id: `call_steer_${index}`,
                name: "lookup",
                arguments: "{}",
                status: "completed",
              }
            : {
                type: "message",
                id: "msg_steer_final",
                role: "assistant",
                status: "completed",
                content: [{ type: "output_text", text: "Done.", annotations: [] }],
              };
        return new Response(
          [
            { type: "response.output_item.done", output_index: 0, item: reasoning },
            { type: "response.output_item.done", output_index: 1, item: output },
            {
              type: "response.completed",
              response: {
                id: `resp_steer_${index}`,
                model: model.id,
                status: "completed",
                output: [reasoning, output],
                usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
              },
            },
          ]
            .map((event) => `data: ${JSON.stringify(event)}\n\n`)
            .join(""),
          {
            headers: { "content-type": "text/event-stream" },
          },
        );
      },
    });
    streamMocks.streamSimple.mockImplementation((requestModel, context, options) => {
      const requestOptions = {
        ...options,
        onPayload: (payload: unknown) => Response.json(payload).json(),
      };
      return entry === "managed"
        ? createAzureOpenAIResponsesTransportStreamFn()(requestModel, context, requestOptions)
        : streamAzureOpenAIResponses(requestModel, context, requestOptions);
    });
    await withOpenClawTestState({ label: "responses-steering-cache" }, async (state) => {
      const target = {
        agentId: "main",
        sessionId: "steering-cache",
        sessionKey: "agent:main:steering-cache",
        storePath: path.join(state.agentDir("main"), "openclaw-agent.sqlite"),
      };
      await upsertSessionEntryCore(target, { sessionId: target.sessionId, updatedAt: 1 });
      const manager = await SessionManager.openAsync(target, state.workspaceDir);
      let toolCalls = 0;
      ({ session } = await createTestSession({
        model,
        sessionManager: manager,
        customTools: [
          {
            name: "lookup",
            label: "Lookup",
            description: "Lookup fixture.",
            parameters: Type.Object({}),
            execute: async () => {
              if (++toolCalls === 1) {
                await session.steer(
                  "First steering question.",
                  undefined,
                  undefined,
                  undefined,
                  undefined,
                  undefined,
                  undefined,
                  { text: "Original permission and conversation facts." },
                );
              }
              return { content: [{ type: "text", text: "found" }], details: {} };
            },
          },
        ],
      }));
      const baseConvert = session.agent.convertToLlm.bind(session.agent);
      session.agent.convertToLlm = async (messages) => {
        const converted = await baseConvert(
          normalizeMessagesForLlmBoundary(messages, {
            sessionVersion: 4,
            includeTimestamp: false,
          }),
        );
        for (const message of converted) {
          if (isRuntimeContextMessage(message)) {
            setRuntimeContextRetention(message, false);
          }
        }
        return converted;
      };
      const carrier = buildRuntimeContextCustomMessage(
        "Original permission and conversation facts.",
      );
      assert(carrier);
      const cleanup = installRuntimeContextMessageForPrompt({ session, message: carrier });
      try {
        await session.prompt("Original question.");
      } finally {
        cleanup();
      }
      expect(requests).toHaveLength(4);
      const withoutPoints = (items: WireItem[]) =>
        items.map((item) => ({
          ...item,
          ...(item.content
            ? {
                content: item.content.map(({ prompt_cache_breakpoint: _point, ...block }) => block),
              }
            : {}),
        }));
      const isCarrier = (item: WireItem) =>
        item.role === "developer" &&
        item.content?.some(
          (block) =>
            typeof block.text === "string" &&
            block.text.includes("permission and conversation facts"),
        );
      for (let index = 1; index < requests.length; index++) {
        const previous = requests[index - 1];
        const current = requests[index];
        assert(previous && current);
        const previousHistory = previous.filter((item) => !isCarrier(item));
        const currentHistory = current.filter((item) => !isCarrier(item));
        expect(withoutPoints(currentHistory.slice(0, previousHistory.length))).toEqual(
          withoutPoints(previousHistory),
        );
        const currentCarriers = current.filter(isCarrier);
        expect(current.slice(-currentCarriers.length)).toEqual(currentCarriers);
        expect(currentCarriers.slice(0, previous.filter(isCarrier).length)).toEqual(
          previous.filter(isCarrier),
        );
      }
      const beforeFinal = requests[2];
      const final = requests[3];
      assert(beforeFinal && final);
      expect(final.filter(isCarrier)).toEqual(beforeFinal.filter(isCarrier));
      expect(
        requests.map(
          (items) =>
            items.filter(
              (item) =>
                item.role === "developer" &&
                item.content?.some(
                  (block) =>
                    typeof block.text === "string" &&
                    block.text.includes("permission and conversation facts"),
                ),
            ).length,
        ),
      ).toEqual([1, 2, 3, 3]);
      for (const [index, items] of requests.entries()) {
        const firstCarrierIndex = items.findIndex(isCarrier);
        const pointIndex = items.findIndex((item) =>
          item.content?.some((block) => block.prompt_cache_breakpoint),
        );
        expect(pointIndex).toBeLessThan(firstCarrierIndex);
        if (index === 3) {
          expect(pointIndex).toBeLessThan(firstCarrierIndex - 1);
        }
        const points = items.flatMap(
          (item) => item.content?.filter((block) => block.prompt_cache_breakpoint) ?? [],
        );
        expect(points).toEqual([
          {
            type: "input_text",
            text:
              index === 0
                ? "Original question."
                : index === 1
                  ? "First steering question."
                  : "Second steering question.",
            prompt_cache_breakpoint: { mode: "explicit" },
          },
        ]);
      }
      const reopened = await SessionManager.openAsync(target, state.workspaceDir);
      expect(
        convertToLlm(
          normalizeMessagesForLlmBoundary(reopened.buildSessionContext().messages, {
            sessionVersion: 4,
            includeTimestamp: false,
          }),
        ).filter(isRuntimeContextMessage),
      ).toEqual([]);
      const nextCarrier = buildRuntimeContextCustomMessage(
        "Next turn permission and conversation facts.",
      );
      assert(nextCarrier);
      const nextCleanup = installRuntimeContextMessageForPrompt({
        session,
        message: nextCarrier,
      });
      try {
        await session.prompt("Independent question.");
      } finally {
        nextCleanup();
      }
      expect(requests).toHaveLength(5);
      const next = requests[4];
      assert(next);
      expect(next.filter(isCarrier)).toEqual([
        expect.objectContaining({
          role: "developer",
          content: [
            expect.objectContaining({
              text: expect.stringContaining("Next turn permission and conversation facts."),
            }),
          ],
        }),
      ]);
      expect(next.at(-1)).toEqual(next.find(isCarrier));
      expect(
        next.flatMap(
          (item) => item.content?.filter((block) => block.prompt_cache_breakpoint) ?? [],
        ),
      ).toEqual([
        {
          type: "input_text",
          text: "Independent question.",
          prompt_cache_breakpoint: { mode: "explicit" },
        },
      ]);
    });
  },
);
