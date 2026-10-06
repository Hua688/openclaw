import { createHash } from "node:crypto";
import { chmod, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getAiTransportHost } from "../../../../packages/ai/src/host.js";
import { createAzureOpenAIResponsesTransportStreamFn } from "../../../../packages/ai/src/transports/openai-responses-client.js";
import { useAutoCleanupTempDirTracker } from "../../../../test/helpers/temp-dir.js";
import type { Model } from "../../../llm/types.js";
import { runEmbeddedAttemptPromptPhase } from "./attempt-prompt-phase.js";

const { createFixture, mocks } = await vi.hoisted(
  async () => await import("./attempt-prompt-phase.test-support.js"),
);
const tempDirs = useAutoCleanupTempDirTracker(afterEach);
const scopeSymbol = Symbol.for("openclaw.azureResponsesCacheCaptureScope.v1");
const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("prompt-phase capture activation through the Azure SDK", () => {
  it.each(["default-off", "matching", "other-session", "other-api"] as const)(
    "captures only selected non-compaction requests (%s)",
    async (mode) => {
      const fixture = createFixture();
      const { attempt, prepared } = fixture.input;
      const session = prepared.sessionRuntime.agentSession.activeSession;
      const directory = tempDirs.make("openclaw-prompt-capture-");
      await chmod(directory, 0o700);
      await writeFile(path.join(directory, ".enabled"), "", { mode: 0o600 });
      vi.stubGlobal(
        scopeSymbol,
        mode === "default-off"
          ? undefined
          : {
              directory,
              sessionHash: sha256(mode === "other-session" ? "another-session" : attempt.sessionId),
            },
      );
      const model = {
        id: "capture-model",
        name: "Capture fixture",
        api: "azure-openai-responses",
        provider: "azure",
        baseUrl: "https://capture.invalid/openai/v1",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 200_000,
        maxTokens: 1024,
      } satisfies Model<"azure-openai-responses">;
      attempt.model = model;
      const fetch = vi.fn<typeof globalThis.fetch>().mockImplementation(async () => {
        const event = {
          type: "response.completed",
          response: {
            id: "resp_synthetic",
            status: "completed",
            output: [],
            usage: {
              input_tokens: 120,
              output_tokens: 0,
              total_tokens: 120,
              input_tokens_details: { cached_tokens: 80, cache_write_tokens: 40 },
            },
          },
        };
        return new Response(`data: ${JSON.stringify(event)}\n\n`, {
          headers: { "content-type": "text/event-stream" },
        });
      });
      vi.spyOn(getAiTransportHost(), "buildModelFetch").mockReturnValue(fetch);
      session.agent.streamFn = createAzureOpenAIResponsesTransportStreamFn();
      let compacting = false;
      Object.defineProperty(session, "isCompacting", { get: () => compacting });
      mocks.applyPromptToolsAllow.mockReturnValue(prepared.promptToolPolicy.current);
      mocks.submitPrompt.mockImplementation(async () => {
        for (const index of [0, 1, 2]) {
          compacting = index === 1;
          if (index === 2) {
            prepared.promptToolPolicy.current.activeToolNames.push("screen");
          }
          const result = await (
            await session.agent.streamFn(
              mode === "other-api" ? { ...model, api: "openai-responses" } : model,
              { systemPrompt: "## Skills\nsynthetic skill", messages: [], tools: [] },
              { apiKey: "synthetic-key", sessionId: attempt.sessionId },
            )
          ).result();
          expect(result.errorMessage).toBeUndefined();
        }
      });
      await runEmbeddedAttemptPromptPhase(fixture.input, fixture.promptState);
      expect(fixture.readState().promptError).toBeNull();
      expect(fetch).toHaveBeenCalledTimes(3);
      if (mode !== "matching") {
        expect(await readdir(directory)).toEqual([".enabled"]);
        return;
      }
      const runDirectory = path.join(directory, sha256(attempt.sessionId), sha256(attempt.runId));
      expect(await readdir(runDirectory)).toEqual(["1", "2"]);
      for (const index of [1, 2]) {
        const requestDirectory = path.join(runDirectory, String(index));
        const attempts = await readdir(requestDirectory);
        expect(attempts).toHaveLength(1);
        const [attemptId] = attempts;
        if (!attemptId) {
          throw new Error("expected one captured SDK attempt");
        }
        const identity = {
          sessionHash: sha256(attempt.sessionId),
          runHash: sha256(attempt.runId),
          requestIndex: index,
          uniqueAttemptId: attemptId,
        };
        const attemptDirectory = path.join(requestDirectory, attemptId);
        const dispatch = JSON.parse(
          await readFile(path.join(attemptDirectory, "dispatch-intent.json"), "utf8"),
        );
        const http = JSON.parse(
          await readFile(path.join(attemptDirectory, "http-response.json"), "utf8"),
        );
        const terminal = JSON.parse(
          await readFile(path.join(attemptDirectory, "selected-terminal.json"), "utf8"),
        );
        const body = await readFile(path.join(attemptDirectory, "request-body.bin"));
        const sent = fetch.mock.calls[index === 1 ? 0 : 2]?.[1]?.body;
        if (!(sent instanceof Uint8Array)) {
          throw new Error("expected final SDK encoded bytes");
        }
        expect(body).toEqual(Buffer.from(sent));
        expect(dispatch).toMatchObject({
          ...identity,
          bodyByteLength: body.length,
          bodySha256: createHash("sha256").update(body).digest("hex"),
          cacheTrackingFacts: { tools: { promptPolicy: { activeToolCount: index } } },
        });
        expect(http).toMatchObject({ ...identity, status: 200, ok: true });
        expect(terminal).toMatchObject({
          ...identity,
          terminalEventType: "response.completed",
          usage: { input_tokens_details: { cached_tokens: 80, cache_write_tokens: 40 } },
        });
      }
    },
  );
});
