import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { labelRuntimeContextText } from "../../../llm/types.js";
import { Agent } from "../../runtime/index.js";
import { guardSessionManager } from "../../session-tool-result-guard-wrapper.js";
import { convertToLlm } from "../../sessions/messages.js";
import { SessionManager } from "../../sessions/session-manager.js";
import { prepareEmbeddedAttemptSessionBoundary } from "./attempt-session-prepare.js";
import { buildRuntimeContextCustomMessage } from "./runtime-context-prompt.js";

describe("private runtime-context normalization observations", () => {
  it.each([false, true])(
    "records typed converted retention without wire-placement claims (%s)",
    async (retained) => {
      const agent = new Agent({ convertToLlm });
      const boundary = await prepareEmbeddedAttemptSessionBoundary({
        activeSession: { agent },
        appendOnlyRuntimeContext: retained,
        captureNormalizationFacts: true,
        attempt: { sessionId: "capture-boundary", prompt: "question" },
        getUserTranscriptContexts: () => undefined,
        isRawModelRun: false,
        preparedUserTurnMessage: undefined,
        sessionManager: guardSessionManager(SessionManager.inMemory()),
        setActiveSessionSystemPrompt: vi.fn(),
      });
      const user = { role: "user" as const, content: "question", timestamp: 1 };
      const carrier = buildRuntimeContextCustomMessage("synthetic context");
      if (!carrier) {
        throw new Error("Expected synthetic runtime context");
      }
      const converted = await agent.convertToLlm(retained ? [user, carrier] : [carrier, user]);
      const text = labelRuntimeContextText("synthetic context");
      expect(converted.at(-1)).toMatchObject({ runtimeContext: { retained }, content: text });
      expect(boundary.getCacheTrackingFacts?.()).toMatchObject({
        status: "captured",
        appendOnlyPolicy: retained,
        wirePlacement: { status: "unavailable", reason: "provider-egress-owned" },
        stages: {
          input: { count: 1, carriers: [{ kind: "custom-carrier" }] },
          normalized: { count: 1 },
          converted: {
            count: 1,
            carriers: [
              {
                index: converted.length - 1,
                kind: "user-carrier",
                retention: { status: "captured", retained },
                contentChars: text.length,
                contentSha256: createHash("sha256").update(text).digest("hex"),
              },
            ],
          },
        },
      });
      expect(JSON.stringify(converted)).not.toContain("provider-egress-owned");
    },
  );
});
