import { describe, expect, it } from "vitest";
import { Agent } from "../../packages/agent-core/src/agent.js";
import { convertToLlm as convertAgentTranscriptToLlm } from "../../packages/agent-core/src/harness/messages.js";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type Context,
  type Model,
} from "../../packages/agent-core/src/llm.js";
import type { AgentMessage, StreamFn } from "../../packages/agent-core/src/types.js";
import { installRuntimeContextMessageForPrompt } from "./embedded-agent-runner/run/attempt-llm-boundary.js";
import { prepareEmbeddedAttemptSessionBoundary } from "./embedded-agent-runner/run/attempt-session-prepare.js";
import { buildRuntimeContextCustomMessage } from "./embedded-agent-runner/run/runtime-context-prompt.js";
import { guardSessionManager } from "./session-tool-result-guard-wrapper.js";
import { SessionManager } from "./sessions/session-manager.js";

const model = {
  id: "test-model",
  name: "Test Model",
  api: "test-api",
  provider: "test-provider",
  baseUrl: "https://example.test",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1000,
  maxTokens: 1000,
} satisfies Model;

const testUsage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

describe("runtime context with queued follow-ups", () => {
  it("keeps runtime context in the submitted prefix through an agent-end plan-check", async () => {
    const runtimeContext = buildRuntimeContextCustomMessage("runtime context");
    if (!runtimeContext) {
      throw new Error("Expected a runtime-context carrier");
    }

    const planCheck: AgentMessage = {
      role: "custom",
      customType: "openclaw.plan-completion-check",
      content: "check the unfinished plan",
      display: false,
      timestamp: 2,
    };
    const requestContexts: Context[] = [];
    const streamFn: StreamFn = (activeModel, context) => {
      requestContexts.push(structuredClone(context));
      const stream = createAssistantMessageEventStream();
      const requestNumber = requestContexts.length;
      queueMicrotask(() => {
        const message: AssistantMessage = {
          role: "assistant",
          content: [{ type: "text", text: `answer ${requestNumber}` }],
          api: activeModel.api,
          provider: activeModel.provider,
          model: activeModel.id,
          usage: testUsage,
          stopReason: "stop",
          timestamp: requestNumber,
        };
        stream.push({ type: "done", reason: "stop", message });
        stream.end();
      });
      return stream;
    };
    const agent = new Agent({
      initialState: {
        model,
        systemPrompt: "",
        tools: [],
        messages: [],
      },
      convertToLlm: (messages: AgentMessage[]) => convertAgentTranscriptToLlm(messages),
      streamFn,
    });
    await prepareEmbeddedAttemptSessionBoundary({
      activeSession: { agent },
      appendOnlyRuntimeContext: false,
      attempt: { prompt: "original request", trigger: "user" },
      getUserTranscriptContexts: () => undefined,
      isRawModelRun: false,
      preparedUserTurnMessage: undefined,
      sessionManager: guardSessionManager(SessionManager.inMemory(), {
        runId: "runtime-context-follow-up",
      }),
      setActiveSessionSystemPrompt: () => undefined,
    });
    const session = {
      agent,
      get messages() {
        return agent.state.messages;
      },
    };
    const cleanupRuntimeContext = installRuntimeContextMessageForPrompt({
      session,
      message: runtimeContext,
    });
    let queuedPlanCheck = false;
    agent.subscribe((event) => {
      if (event.type === "agent_end" && !queuedPlanCheck) {
        queuedPlanCheck = true;
        agent.followUp(planCheck);
      }
    });

    try {
      await agent.prompt("original request");
      expect(queuedPlanCheck).toBe(true);
      await agent.continue();
    } finally {
      cleanupRuntimeContext();
    }

    expect(requestContexts).toHaveLength(2);
    const firstMessages = requestContexts[0]?.messages ?? [];
    const followUpMessages = requestContexts[1]?.messages ?? [];
    expect(firstMessages.map((message) => message.role)).toEqual(["user", "user"]);
    expect(firstMessages[0]).toMatchObject({
      role: "user",
      content: "original request",
    });
    expect(firstMessages[1]).toMatchObject({
      role: "user",
      content: [{ type: "text", text: expect.stringContaining("runtime context") }],
    });
    expect(followUpMessages.map((message) => message.role)).toEqual([
      "user",
      "user",
      "assistant",
      "user",
    ]);
    expect(followUpMessages[2]).toMatchObject({
      role: "assistant",
      content: [{ type: "text", text: "answer 1" }],
    });
    expect(followUpMessages[3]).toMatchObject({
      role: "user",
      content: [{ type: "text", text: "check the unfinished plan" }],
    });
    expect(followUpMessages.slice(0, firstMessages.length)).toEqual(firstMessages);
  });
});
