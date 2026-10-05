import { createHash } from "node:crypto";
import { chmod, mkdir, readFile, readdir, rename, symlink, writeFile } from "node:fs/promises";
import * as fs from "node:fs/promises";
import path from "node:path";
import type { Model } from "@openclaw/llm-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAutoCleanupTempDirTracker } from "../../../../test/helpers/temp-dir.js";

const transport = vi.hoisted(() => ({
  fetch: vi.fn<typeof fetch>(),
  warn: vi.fn(),
}));
vi.mock("node:fs/promises", async (original) => {
  const actual = await original<typeof import("node:fs/promises")>();
  return { ...actual, lstat: vi.fn(actual.lstat) };
});
vi.mock("./host-policy.js", async (original) => ({
  ...(await original<typeof import("./host-policy.js")>()),
  buildGuardedModelFetch: () => transport.fetch,
}));
vi.mock("./openai-transport-shared.js", async (original) => {
  const actual = await original<typeof import("./openai-transport-shared.js")>();
  return { ...actual, log: { ...actual.log, warn: transport.warn } };
});

import {
  AZURE_RESPONSES_CAPTURE_CONTEXT,
  createAzureResponsesCaptureFetch,
} from "./azure-responses-cache-capture.js";
import { createAzureOpenAIResponsesTransportStreamFn } from "./openai-responses-client.js";

const tempDirs = useAutoCleanupTempDirTracker(afterEach);
const model = {
  id: "capture-model",
  name: "Capture fixture",
  api: "azure-openai-responses",
  provider: "azure",
  baseUrl: "https://capture.invalid/openai/v1",
  reasoning: true,
  input: ["text"],
  cost: { input: 1, output: 1, cacheRead: 1, cacheWrite: 1 },
  contextWindow: 200_000,
  maxTokens: 1024,
} satisfies Model<"azure-openai-responses">;

function completed(input: number, read: number, written: number): Response {
  const event = {
    type: "response.completed",
    response: {
      id: "resp_synthetic",
      status: "completed",
      output: [
        {
          id: "msg_synthetic",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "ok", annotations: [] }],
        },
      ],
      usage: {
        input_tokens: input,
        output_tokens: 2,
        total_tokens: input + 2,
        input_tokens_details: { cached_tokens: read, cache_write_tokens: written },
      },
    },
  };
  return new Response(`data: ${JSON.stringify(event)}\n\n`, {
    status: 200,
    headers: { "content-type": "text/event-stream", "x-request-id": "synthetic-request" },
  });
}

describe("Azure final HTTP request capture", () => {
  beforeEach(() => {
    transport.fetch.mockReset();
    transport.warn.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  async function scope(marker: boolean) {
    const directory = tempDirs.make("openclaw-capture-");
    await chmod(directory, 0o700);
    const enabledFile = path.join(directory, ".enabled");
    if (marker) {
      await writeFile(enabledFile, "", { mode: 0o600 });
    }
    return {
      directory,
      enabledFile,
      sessionHash: "a".repeat(64),
      runHash: "b".repeat(64),
      requestIndex: 1,
      minimumFreeBytes: 0,
      cacheTrackingFacts: () => ({ syntheticFact: true }),
    };
  }

  async function run(capture: Awaited<ReturnType<typeof scope>>, extra = "first") {
    const options = Object.assign(
      {
        apiKey: "synthetic-key",
        sessionId: "synthetic-session",
        transport: "sse" as const,
        onPayload: (payload: unknown) => {
          if (!payload || typeof payload !== "object") {
            throw new Error("expected request payload");
          }
          return { ...payload, metadata: { synthetic: extra } };
        },
      },
      { [AZURE_RESPONSES_CAPTURE_CONTEXT]: capture },
    );
    const result = await (
      await createAzureOpenAIResponsesTransportStreamFn()(
        model,
        {
          systemPrompt:
            extra === "second"
              ? "## Skills\nsynthetic skill"
              : "## Skills\nsynthetic skill\n## Deferred Tooling\nscreen",
          messages: [],
          tools: [],
        },
        options,
      )
    ).result();
    expect(result.errorMessage).toBeUndefined();
    return result;
  }

  it("keeps final payload bytes unchanged and creates no capture without the marker", async () => {
    const capture = await scope(false);
    transport.fetch.mockResolvedValueOnce(completed(100, 60, 40));
    const result = await run(capture);
    expect(result.stopReason).not.toBe("error");
    expect(await readdir(capture.directory)).toEqual([]);
    expect(transport.fetch).toHaveBeenCalledTimes(1);
    const sent = transport.fetch.mock.calls[0]?.[1]?.body;
    if (!(sent instanceof Uint8Array)) {
      throw new Error("expected final SDK encoded bytes");
    }
    expect(JSON.parse(new TextDecoder().decode(sent))).toMatchObject({
      metadata: { synthetic: "first" },
    });
    expect(transport.warn).not.toHaveBeenCalled();
  });

  it("joins actual dispatched bytes, HTTP and selected terminal usage per unique attempt", async () => {
    const capture = await scope(true);
    transport.fetch
      .mockResolvedValueOnce(completed(100, 60, 40))
      .mockResolvedValueOnce(completed(120, 80, 40));
    expect((await run(capture)).stopReason).not.toBe("error");
    expect((await run(capture, "second")).stopReason).not.toBe("error");
    const requestDirectory = path.join(
      capture.directory,
      capture.sessionHash,
      capture.runHash,
      "1",
    );
    const attempts = await readdir(requestDirectory);
    expect(attempts).toHaveLength(2);
    const capturedBodies: string[] = [];
    for (const attemptId of attempts) {
      const directory = path.join(requestDirectory, attemptId);
      const bytes = await readFile(path.join(directory, "request-body.bin"));
      capturedBodies.push(bytes.toString());
      const dispatch = JSON.parse(
        await readFile(path.join(directory, "dispatch-intent.json"), "utf8"),
      );
      const http = JSON.parse(await readFile(path.join(directory, "http-response.json"), "utf8"));
      const terminal = JSON.parse(
        await readFile(path.join(directory, "selected-terminal.json"), "utf8"),
      );
      const identity = {
        sessionHash: capture.sessionHash,
        runHash: capture.runHash,
        requestIndex: 1,
        uniqueAttemptId: attemptId,
      };
      expect(dispatch).toMatchObject({
        ...identity,
        bodyByteLength: bytes.length,
        bodySha256: createHash("sha256").update(bytes).digest("hex"),
        cacheTrackingFacts: { syntheticFact: true },
        requestSurface: {
          state: "captured",
          deferredToolDirectory:
            terminal.usage.input_tokens === 100
              ? { screen: true, suggestTask: false, dismissTask: false }
              : { status: "unavailable", reason: "heading-not-captured" },
        },
      });
      expect(http).toMatchObject({ ...identity, status: 200, ok: true });
      expect(terminal).toMatchObject({
        ...identity,
        state: "selected_response_terminal",
        terminalEventType: "response.completed",
        usage: { input_tokens_details: { cache_write_tokens: 40 } },
      });
      expect(terminal.usage.input_tokens_details.cached_tokens).toBe(
        terminal.usage.input_tokens === 100 ? 60 : 80,
      );
    }
    expect(capturedBodies.toSorted()).toEqual(
      transport.fetch.mock.calls
        .map(([, init]) => {
          if (!(init?.body instanceof Uint8Array)) {
            throw new Error("expected final SDK encoded bytes");
          }
          return new TextDecoder().decode(init.body);
        })
        .toSorted(),
    );
    expect(transport.warn).not.toHaveBeenCalled();
  });

  it("reports an unsafe marker without changing or blocking the provider request", async () => {
    const capture = await scope(false);
    await mkdir(capture.enabledFile);
    transport.fetch.mockResolvedValueOnce(completed(100, 60, 40));
    expect((await run(capture)).stopReason).not.toBe("error");
    expect(transport.fetch).toHaveBeenCalledTimes(1);
    expect(await readdir(capture.directory)).toEqual([".enabled"]);
    expect(transport.warn).toHaveBeenCalledWith(
      expect.stringContaining("operation=capture_control"),
    );
  });

  it("makes EACCES visible without disclosing the body or suppressing the provider result", async () => {
    const capture = await scope(true);
    vi.mocked(fs.lstat).mockRejectedValueOnce(
      Object.assign(new Error("private-path"), { code: "EACCES" }),
    );
    transport.fetch.mockResolvedValueOnce(completed(100, 60, 40));
    expect((await run(capture)).stopReason).not.toBe("error");
    expect(transport.warn).toHaveBeenCalledWith(expect.stringContaining("code=EACCES"));
    expect(JSON.stringify(transport.warn.mock.calls)).not.toContain("synthetic skill");
    expect(await readdir(capture.directory)).toEqual([".enabled"]);
  });

  it.each(["http", "terminal"] as const)(
    "rejects a late ancestor symlink before %s sidecar writes without disrupting SSE",
    async (stage) => {
      const capture = await scope(true);
      const runDirectory = path.join(capture.directory, capture.sessionHash, capture.runHash);
      const savedRun = `${runDirectory}.saved`;
      const redirectedRun = path.join(tempDirs.make("openclaw-capture-redirect-"), "run");
      let redirectedAttempt: string | undefined;
      let savedAttempt: string | undefined;
      const redirect = async () => {
        const attempts = await readdir(path.join(runDirectory, "1"));
        expect(attempts).toHaveLength(1);
        const [attemptId] = attempts;
        if (!attemptId) {
          throw new Error("expected a dispatched SDK attempt");
        }
        await rename(runDirectory, savedRun);
        redirectedAttempt = path.join(redirectedRun, "1", attemptId);
        savedAttempt = path.join(savedRun, "1", attemptId);
        await mkdir(redirectedAttempt, { recursive: true, mode: 0o700 });
        await symlink(
          redirectedRun,
          runDirectory,
          process.platform === "win32" ? "junction" : "dir",
        );
      };
      const frame = await completed(100, 60, 40).text();
      transport.fetch.mockImplementationOnce(async () => {
        if (stage === "http") {
          await redirect();
          return completed(100, 60, 40);
        }
        return new Response(
          new ReadableStream(
            {
              async pull(controller) {
                await redirect();
                controller.enqueue(new TextEncoder().encode(frame));
                controller.close();
              },
            },
            { highWaterMark: 0 },
          ),
          { headers: { "content-type": "text/event-stream" } },
        );
      });
      const result = await run(capture);
      expect(result.stopReason).not.toBe("error");
      expect(result.usage.cacheRead).toBe(60);
      if (!redirectedAttempt || !savedAttempt) {
        throw new Error("expected a late directory replacement");
      }
      expect(await readdir(redirectedAttempt)).toEqual([]);
      const savedFiles = await readdir(savedAttempt);
      expect(savedFiles).toEqual(
        expect.arrayContaining(["dispatch-intent.json", "request-body.bin"]),
      );
      expect(savedFiles).not.toContain("selected-terminal.json");
      expect(savedFiles.includes("http-response.json")).toBe(stage === "terminal");
      expect(transport.warn).toHaveBeenCalledWith(
        expect.stringContaining(
          `operation=${stage === "http" ? "http_response" : "selected_terminal"}`,
        ),
      );
    },
  );

  it("preserves fetch input/init identity and the exact underlying error", async () => {
    const capture = await scope(true);
    const failure = new TypeError("synthetic fetch failure");
    transport.fetch.mockRejectedValueOnce(failure);
    const fetch = createAzureResponsesCaptureFetch(transport.fetch, capture, new WeakMap());
    const input = new URL("https://capture.invalid/v1/responses");
    const init = { method: "POST", body: '{"synthetic":true}', headers: { "x-synthetic": "one" } };
    await expect(fetch(input, init)).rejects.toBe(failure);
    const call = transport.fetch.mock.calls[0];
    if (!call) {
      throw new Error("expected provider fetch");
    }
    expect(call[0]).toBe(input);
    expect(call[1]).toBe(init);
    expect(init).toEqual({
      method: "POST",
      body: '{"synthetic":true}',
      headers: { "x-synthetic": "one" },
    });
    const requestDirectory = path.join(
      capture.directory,
      capture.sessionHash,
      capture.runHash,
      "1",
    );
    const [attempt] = await readdir(requestDirectory);
    if (!attempt) {
      throw new Error("expected a captured fetch attempt");
    }
    const directory = path.join(requestDirectory, attempt);
    expect(await readdir(directory)).toEqual([
      "dispatch-intent.json",
      "fetch-error.json",
      "request-body.bin",
    ]);
    expect(
      JSON.parse(await readFile(path.join(directory, "fetch-error.json"), "utf8")),
    ).toMatchObject({
      state: "fetch_error",
      sessionHash: capture.sessionHash,
      runHash: capture.runHash,
      requestIndex: 1,
      uniqueAttemptId: attempt,
      errorName: "TypeError",
    });
  });
});
