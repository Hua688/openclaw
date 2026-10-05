import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, open, rename, rm, statfs } from "node:fs/promises";
import path from "node:path";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import { log } from "./openai-transport-shared.js";

export const AZURE_RESPONSES_CAPTURE_CONTEXT = Symbol.for(
  "openclaw.azureResponsesCacheCaptureContext.v1",
);

export type AzureResponsesCaptureContext = {
  sessionHash: string;
  runHash: string;
  requestIndex: number;
  directory: string;
  enabledFile: string;
  minimumFreeBytes: number;
  cacheTrackingFacts?: () => unknown;
};

type CaptureAttempt = {
  context: AzureResponsesCaptureContext;
  attemptId: string;
  directory: string;
};
const IO_TIMEOUT_MS = 10_000;
const SIDECAR_RESERVE_BYTES = 64 * 1024;
let pendingCaptureBytes = 0;

function digest(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function captureIdentity(attempt: CaptureAttempt) {
  return {
    sessionHash: attempt.context.sessionHash,
    runHash: attempt.context.runHash,
    requestIndex: attempt.context.requestIndex,
    uniqueAttemptId: attempt.attemptId,
  };
}

export function readAzureResponsesCaptureContext(
  options: unknown,
): AzureResponsesCaptureContext | undefined {
  if (!options || typeof options !== "object") {
    return undefined;
  }
  const value: unknown = Reflect.get(options, AZURE_RESPONSES_CAPTURE_CONTEXT);
  if (!isRecord(value)) {
    return undefined;
  }
  if (
    typeof value.sessionHash !== "string" ||
    !/^[a-f\d]{64}$/i.test(value.sessionHash) ||
    typeof value.runHash !== "string" ||
    !/^[a-f\d]{64}$/i.test(value.runHash) ||
    typeof value.requestIndex !== "number" ||
    !Number.isSafeInteger(value.requestIndex) ||
    value.requestIndex < 1 ||
    typeof value.directory !== "string" ||
    !path.isAbsolute(value.directory) ||
    typeof value.enabledFile !== "string" ||
    path.resolve(value.enabledFile) !== path.join(path.resolve(value.directory), ".enabled") ||
    typeof value.minimumFreeBytes !== "number" ||
    !Number.isSafeInteger(value.minimumFreeBytes) ||
    value.minimumFreeBytes < 0
  ) {
    log.warn("[responses-cache-capture] invalid_context");
    return undefined;
  }
  const tracking = value.cacheTrackingFacts;
  return {
    sessionHash: value.sessionHash,
    runHash: value.runHash,
    requestIndex: value.requestIndex,
    directory: path.resolve(value.directory),
    enabledFile: path.resolve(value.enabledFile),
    minimumFreeBytes: value.minimumFreeBytes,
    ...(typeof tracking === "function"
      ? { cacheTrackingFacts: () => Reflect.apply(tracking, undefined, []) as unknown }
      : {}),
  };
}

function report(attempt: CaptureAttempt, operation: string, error?: unknown): void {
  const errorName = error instanceof Error ? error.name : error === undefined ? "none" : "unknown";
  const errorCode =
    isRecord(error) && typeof error.code === "string" && /^[A-Z_]+$/.test(error.code)
      ? ` code=${error.code}`
      : "";
  log.warn(
    `[responses-cache-capture] coverage_gap sessionHash=${attempt.context.sessionHash} requestIndex=${attempt.context.requestIndex} attemptId=${attempt.attemptId} operation=${operation} error=${errorName}${errorCode}`,
  );
}

async function withDeadline<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  parent?: AbortSignal,
): Promise<T> {
  parent?.throwIfAborted();
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let removeAbort: (() => void) | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error("Azure Responses capture I/O timed out");
      error.name = "AzureResponsesCaptureIoTimeoutError";
      controller.abort(error);
      reject(error);
    }, IO_TIMEOUT_MS);
  });
  const aborted = new Promise<never>((_, reject) => {
    if (!parent) {
      return;
    }
    const onAbort = () => {
      const reason: unknown = parent.reason;
      controller.abort(reason);
      reject(
        reason instanceof Error
          ? reason
          : new Error("Azure Responses capture I/O aborted", { cause: reason }),
      );
    };
    parent.addEventListener("abort", onAbort, { once: true });
    removeAbort = () => parent.removeEventListener("abort", onAbort);
    if (parent.aborted) {
      onAbort();
    }
  });
  try {
    return await Promise.race([
      Promise.resolve().then(() => operation(controller.signal)),
      timeout,
      aborted,
    ]);
  } finally {
    clearTimeout(timer);
    removeAbort?.();
  }
}

async function privateDirectory(directory: string, signal: AbortSignal): Promise<void> {
  const info = await lstat(directory);
  signal.throwIfAborted();
  if (
    !info.isDirectory() ||
    info.isSymbolicLink() ||
    (process.platform !== "win32" && (info.mode & 0o777) !== 0o700)
  ) {
    throw new Error("capture directory is not private");
  }
}

async function enabled(
  context: AzureResponsesCaptureContext,
  signal: AbortSignal,
): Promise<boolean> {
  let info;
  try {
    info = await lstat(context.enabledFile);
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
  signal.throwIfAborted();
  await privateDirectory(context.directory, signal);
  if (
    !info.isFile() ||
    info.isSymbolicLink() ||
    (process.platform !== "win32" && (info.mode & 0o777) !== 0o600)
  ) {
    throw new Error("capture enable marker is not private");
  }
  return true;
}

async function write(
  attempt: CaptureAttempt,
  name: string,
  contents: string | Uint8Array,
  signal: AbortSignal,
): Promise<void> {
  const { directory } = attempt;
  for (let current = directory; ; current = path.dirname(current)) {
    await privateDirectory(current, signal);
    if (current === attempt.context.directory) {
      break;
    }
    if (path.dirname(current) === current) {
      throw new Error("capture attempt path escaped its root");
    }
  }
  const finalPath = path.join(directory, name);
  const temporaryPath = path.join(directory, `.${name}.${randomUUID()}.tmp`);
  let file: Awaited<ReturnType<typeof open>> | undefined;
  let temporaryExists = false;
  let published = false;
  try {
    file = await open(temporaryPath, "wx", 0o600);
    temporaryExists = true;
    signal.throwIfAborted();
    await file.writeFile(contents, { signal });
    signal.throwIfAborted();
    if (process.platform !== "win32" && ((await file.stat()).mode & 0o777) !== 0o600) {
      throw new Error("capture file permissions are not private");
    }
    await file.close();
    file = undefined;
    await rename(temporaryPath, finalPath);
    temporaryExists = false;
    published = true;
    signal.throwIfAborted();
  } catch (error) {
    const errors: unknown[] = [error];
    for (const cleanup of [
      ...(file ? [() => file?.close()] : []),
      ...(temporaryExists ? [() => rm(temporaryPath, { force: true })] : []),
      ...(published ? [() => rm(finalPath, { force: true })] : []),
    ]) {
      try {
        await cleanup();
      } catch (cleanupError) {
        errors.push(cleanupError);
      }
    }
    throw errors.length > 1
      ? new AggregateError(errors, "Azure Responses capture write failed")
      : error;
  }
}

async function reserve(
  context: AzureResponsesCaptureContext,
  bytes: number,
  signal: AbortSignal,
): Promise<(() => void) | undefined> {
  if (context.minimumFreeBytes === 0) {
    return () => undefined;
  }
  const info = await statfs(context.directory);
  signal.throwIfAborted();
  const available = info.bavail * info.bsize;
  if (!Number.isFinite(available)) {
    throw new Error("disk reserve check returned invalid free bytes");
  }
  const required = bytes + SIDECAR_RESERVE_BYTES;
  if (available - pendingCaptureBytes - required < context.minimumFreeBytes) {
    log.warn("[responses-cache-capture] low_disk_reserve");
    return undefined;
  }
  pendingCaptureBytes += required;
  let released = false;
  return () => {
    if (!released) {
      released = true;
      pendingCaptureBytes -= required;
    }
  };
}

function snapshot(input: RequestInfo | URL, init?: RequestInit) {
  const body = init?.body;
  if (typeof body === "string") {
    return { bodyType: "string", bytes: new TextEncoder().encode(body) };
  }
  if (body instanceof Uint8Array) {
    return { bodyType: "Uint8Array", bytes: Uint8Array.from(body) };
  }
  if (body instanceof ArrayBuffer) {
    return { bodyType: "ArrayBuffer", bytes: new Uint8Array(body.slice(0)) };
  }
  if (ArrayBuffer.isView(body)) {
    return {
      bodyType: "ArrayBufferView",
      bytes: new Uint8Array(body.buffer, body.byteOffset, body.byteLength).slice(),
    };
  }
  return {
    bodyType:
      body === null || body === undefined
        ? input instanceof Request
          ? "Request"
          : "missing"
        : body instanceof ReadableStream
          ? "ReadableStream"
          : "unsupported",
  };
}

const SECTIONS = new Set([
  "Conversation Context",
  "Subagent Context",
  "Runtime Context",
  "Skills",
  "Messaging",
  "Deferred Tooling",
  "Tool Search",
]);
const TARGET_TOOLS = ["screen", "suggest_task", "dismiss_task"];

function summarize(bytes: Uint8Array): unknown {
  let payload: unknown;
  try {
    payload = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return { state: "unavailable", reason: "body-not-json" };
  }
  if (!isRecord(payload)) {
    return { state: "unavailable", reason: "body-not-object" };
  }
  const texts: string[] = [];
  if (typeof payload.instructions === "string") {
    texts.push(payload.instructions);
  }
  for (const item of Array.isArray(payload.input) ? payload.input : []) {
    if (!isRecord(item) || (item.role !== "developer" && item.role !== "system")) {
      continue;
    }
    if (typeof item.content === "string") {
      texts.push(item.content);
    } else if (Array.isArray(item.content)) {
      const parts = item.content.flatMap((part) =>
        isRecord(part) && typeof part.text === "string" ? [part.text] : [],
      );
      if (parts.length) {
        texts.push(parts.join("\n"));
      }
    }
  }
  const sections: { label: string; chars: number; sha256: string }[] = [];
  let deferred: string | undefined;
  for (const text of texts) {
    const matches = Array.from(text.matchAll(/^## ([^\r\n]+)\r?$/gmu));
    for (const [index, match] of matches.entries()) {
      const label = match[1]?.trim();
      if (!label || !SECTIONS.has(label)) {
        continue;
      }
      const section = text.slice(match.index + match[0].length, matches[index + 1]?.index).trim();
      sections.push({ label, chars: section.length, sha256: digest(section) });
      if (label === "Deferred Tooling") {
        deferred = section;
      }
    }
  }
  const tools: unknown[] = Array.isArray(payload.tools) ? payload.tools : [];
  const nameOf = (tool: unknown) =>
    isRecord(tool)
      ? typeof tool.name === "string"
        ? tool.name
        : isRecord(tool.function) && typeof tool.function.name === "string"
          ? tool.function.name
          : null
      : null;
  const toolDefinitions = Object.fromEntries(
    TARGET_TOOLS.map((name) => {
      const definition = tools.find((tool) => nameOf(tool) === name);
      const serialized = definition === undefined ? undefined : JSON.stringify(definition);
      return [
        name,
        serialized === undefined
          ? { included: false }
          : {
              included: true,
              chars: serialized.length,
              sha256: digest(serialized),
            },
      ];
    }),
  );
  const choice = payload.tool_choice;
  return {
    state: "captured",
    developerPrompt: {
      partCount: texts.length,
      charCount: texts.reduce((n, text) => n + text.length, 0),
      sha256: digest(texts.join("\n\n")),
      parts: texts.map((text) => ({ charCount: text.length, sha256: digest(text) })),
    },
    developerSections: sections,
    deferredToolDirectory:
      deferred === undefined
        ? { status: "unavailable", reason: "heading-not-captured" }
        : {
            screen: deferred.includes("screen"),
            suggestTask: deferred.includes("suggest_task"),
            dismissTask: deferred.includes("dismiss_task"),
          },
    toolDefinitions,
    toolSurface: {
      count: tools.length,
      namedCount: tools.filter((tool) => nameOf(tool) !== null).length,
      namesSha256: digest(JSON.stringify(tools.map(nameOf))),
      toolChoice: {
        state: Object.hasOwn(payload, "tool_choice") ? "present" : "absent",
        ...(typeof choice === "string"
          ? { kind: "string", value: choice }
          : isRecord(choice)
            ? { kind: "object", type: typeof choice.type === "string" ? choice.type : "unknown" }
            : {}),
      },
    },
  };
}

async function dispatch(
  attempt: CaptureAttempt,
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  parent?: AbortSignal,
): Promise<boolean> {
  const body = snapshot(input, init);
  return withDeadline(async (signal) => {
    await privateDirectory(attempt.context.directory, signal);
    const release = await reserve(attempt.context, body.bytes?.byteLength ?? 0, signal);
    if (!release) {
      report(attempt, "disk_reserve");
      return false;
    }
    try {
      await mkdir(attempt.directory, { recursive: true, mode: 0o700 });
      if (body.bytes) {
        await write(attempt, "request-body.bin", body.bytes, signal);
      }
      let facts: unknown;
      try {
        facts = attempt.context.cacheTrackingFacts?.();
      } catch (error) {
        report(attempt, "cache_tracking_facts", error);
      }
      await write(
        attempt,
        "dispatch-intent.json",
        `${JSON.stringify({
          state: "dispatch_intent",
          ...captureIdentity(attempt),
          method: "POST",
          endpoint: "responses",
          bodyType: body.bodyType,
          bodyComparable: Boolean(body.bytes),
          ...(body.bytes
            ? {
                bodyByteLength: body.bytes.byteLength,
                bodySha256: digest(body.bytes),
                requestSurface: summarize(body.bytes),
              }
            : {}),
          ...(facts !== undefined ? { cacheTrackingFacts: facts } : {}),
        })}\n`,
        signal,
      );
      if (!body.bytes) {
        report(attempt, "request_body_incomparable");
      }
      return true;
    } catch (error) {
      try {
        await rm(attempt.directory, { recursive: true, force: true });
      } catch (cleanupError) {
        report(attempt, "dispatch_intent_cleanup_raw_data_may_remain", cleanupError);
      }
      throw error;
    } finally {
      release();
    }
  }, parent);
}

export function createAzureResponsesCaptureFetch(
  baseFetch: typeof globalThis.fetch,
  context: AzureResponsesCaptureContext,
  responseAttempts: WeakMap<Response, CaptureAttempt>,
): typeof globalThis.fetch {
  return async (input, init) => {
    const request = input instanceof Request ? input : undefined;
    const url =
      request?.url ?? (input instanceof URL ? input.href : typeof input === "string" ? input : "");
    let createRequest = false;
    try {
      createRequest =
        (init?.method ?? request?.method ?? "GET").toUpperCase() === "POST" &&
        /\/responses\/?$/.test(new URL(url, "https://openclaw.invalid").pathname);
    } catch {
      // Invalid URLs remain the underlying fetch owner's error.
    }
    if (!createRequest) {
      return baseFetch(input, init);
    }
    const attemptId = randomUUID();
    const attempt: CaptureAttempt = {
      context,
      attemptId,
      directory: path.join(
        context.directory,
        context.sessionHash,
        context.runHash,
        String(context.requestIndex),
        attemptId,
      ),
    };
    const signal = init?.signal === null ? undefined : (init?.signal ?? request?.signal);
    if (signal?.aborted) {
      report(attempt, "request_aborted_before_capture", signal.reason);
      return baseFetch(input, init);
    }
    let captureEnabled = false;
    try {
      captureEnabled = await withDeadline((s) => enabled(context, s), signal);
    } catch (error) {
      report(attempt, "capture_control", error);
    }
    if (!captureEnabled) {
      return baseFetch(input, init);
    }
    let ready = false;
    try {
      ready = await dispatch(attempt, input, init, signal);
    } catch (error) {
      report(attempt, "dispatch_intent", error);
    }
    let response: Response;
    try {
      response = await baseFetch(input, init);
    } catch (error) {
      if (ready) {
        try {
          await withDeadline((s) =>
            write(
              attempt,
              "fetch-error.json",
              `${JSON.stringify({
                state: "fetch_error",
                ...captureIdentity(attempt),
                errorName: error instanceof Error ? error.name : "unknown",
              })}\n`,
              s,
            ),
          );
        } catch (captureError) {
          report(attempt, "fetch_error", captureError);
        }
      }
      throw error;
    }
    if (ready) {
      responseAttempts.set(response, attempt);
      try {
        const providerRequestIds = Object.fromEntries(
          ["x-request-id", "apim-request-id", "x-ms-request-id"]
            .map((name) => [name, response.headers.get(name)])
            .filter(([, value]) => value !== null),
        );
        await withDeadline(
          (s) =>
            write(
              attempt,
              "http-response.json",
              `${JSON.stringify({
                state: "http_response",
                ...captureIdentity(attempt),
                status: response.status,
                ok: response.ok,
                ...(Object.keys(providerRequestIds).length ? { providerRequestIds } : {}),
              })}\n`,
              s,
            ),
          signal,
        );
      } catch (error) {
        report(attempt, "http_response", error);
      }
    }
    return response;
  };
}

function usage(value: unknown): unknown {
  if (!isRecord(value)) {
    return null;
  }
  const count = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x) && x >= 0;
  const result: Record<string, unknown> = {};
  for (const key of ["input_tokens", "output_tokens", "total_tokens"]) {
    if (count(value[key])) {
      result[key] = value[key];
    }
  }
  for (const [source, keys] of [
    ["input_tokens_details", ["cached_tokens", "cache_write_tokens"]],
    ["output_tokens_details", ["reasoning_tokens"]],
  ] as const) {
    const details = value[source];
    if (isRecord(details)) {
      const selected = Object.fromEntries(
        keys.filter((key) => count(details[key])).map((key) => [key, details[key]]),
      );
      if (Object.keys(selected).length) {
        result[source] = selected;
      }
    }
  }
  return Object.keys(result).length ? result : null;
}

export function prepareAzureResponsesCapture(options: unknown, resolveFetch: () => typeof fetch) {
  const context = readAzureResponsesCaptureContext(options);
  if (!context) {
    return undefined;
  }
  const attempts = new WeakMap<Response, CaptureAttempt>();
  return {
    fetch: createAzureResponsesCaptureFetch(resolveFetch(), context, attempts),
    track<T>(response: Response, stream: AsyncIterable<T>, kind: string, signal?: AbortSignal) {
      const attempt = attempts.get(response);
      return attempt ? captureAzureResponsesTerminalUsage(stream, attempt, kind, signal) : stream;
    },
  };
}

export function captureAzureResponsesTerminalUsage<T>(
  stream: AsyncIterable<T>,
  attempt: CaptureAttempt,
  attemptKind: string,
  signal?: AbortSignal,
): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      let terminalObserved = false;
      const persist = async (record: Record<string, unknown>, operation: string) => {
        try {
          await withDeadline(
            (s) =>
              write(
                attempt,
                "selected-terminal.json",
                `${JSON.stringify({
                  ...record,
                  ...captureIdentity(attempt),
                  attemptKind,
                })}\n`,
                s,
              ),
            signal,
          );
        } catch (error) {
          report(attempt, operation, error);
        }
      };
      try {
        for await (const event of stream) {
          if (
            !terminalObserved &&
            isRecord(event) &&
            isRecord(event.response) &&
            ["response.completed", "response.incomplete", "response.failed"].includes(
              String(event.type),
            )
          ) {
            terminalObserved = true;
            await persist(
              {
                state: "selected_response_terminal",
                terminalEventType: event.type,
                ...(typeof event.response.id === "string" &&
                /^[\w-]{1,256}$/.test(event.response.id)
                  ? { responseId: event.response.id }
                  : {}),
                ...(typeof event.response.status === "string"
                  ? { responseStatus: event.response.status }
                  : {}),
                usage: usage(event.response.usage),
              },
              "selected_terminal",
            );
          }
          yield event;
        }
        if (!terminalObserved) {
          await persist(
            { state: "selected_response_terminal_missing" },
            "selected_terminal_missing",
          );
        }
      } catch (error) {
        if (!terminalObserved) {
          await persist(
            {
              state: "selected_response_stream_error",
              errorName: error instanceof Error ? error.name : "unknown",
            },
            "selected_stream_error",
          );
        }
        throw error;
      } finally {
        if (!terminalObserved) {
          report(attempt, "selected_terminal_not_observed");
        }
      }
    },
  };
}
