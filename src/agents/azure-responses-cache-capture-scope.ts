import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import path from "node:path";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import { containsAsciiControlCharacter } from "@openclaw/normalization-core/string-normalization";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { listAgentIds, resolveAgentWorkspaceDir } from "./agent-scope-config.js";

const log = createSubsystemLogger("responses-cache-capture");
const CONTROL_MAX_BYTES = 64 * 1024;
const CAPTURE_DIRECTORY = ".openclaw/azure-responses-cache-capture";
type Selection = Readonly<{ directory: string; sessions: readonly string[] | "all" }>;
let startup: Promise<void> | undefined;
let selections: ReadonlyMap<string, Selection> = new Map();

async function readSelection(workspaceDir: string): Promise<Selection | undefined> {
  const directory = path.join(workspaceDir, ...CAPTURE_DIRECTORY.split("/"));
  let timer: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();
  try {
    return await Promise.race([
      (async () => {
        const control = path.join(directory, ".enabled");
        // Missing control is ordinary default-off, even in an unprovisioned workspace.
        const marker = await lstat(control);
        const root = await lstat(directory);
        const parent = await lstat(path.dirname(directory));
        if (
          !parent.isDirectory() ||
          parent.isSymbolicLink() ||
          !root.isDirectory() ||
          root.isSymbolicLink() ||
          !marker.isFile() ||
          marker.isSymbolicLink() ||
          (process.platform !== "win32" &&
            ((root.mode & 0o777) !== 0o700 || (marker.mode & 0o777) !== 0o600))
        ) {
          throw new Error("unsafe_control");
        }
        const file = await open(control, constants.O_RDONLY | constants.O_NOFOLLOW);
        try {
          const info = await file.stat();
          if (
            !info.isFile() ||
            info.size > CONTROL_MAX_BYTES ||
            (process.platform !== "win32" && (info.mode & 0o777) !== 0o600)
          ) {
            throw new Error("unsafe_control");
          }
          controller.signal.throwIfAborted();
          const bytes = Buffer.alloc(CONTROL_MAX_BYTES + 1);
          const { bytesRead } = await file.read(bytes, 0, bytes.length, 0);
          controller.signal.throwIfAborted();
          if (bytesRead > CONTROL_MAX_BYTES) {
            throw new Error("oversized_control");
          }
          const value: unknown = JSON.parse(bytes.subarray(0, bytesRead).toString("utf8"));
          if (!isRecord(value) || Object.keys(value).length !== 1) {
            throw new Error("invalid_control");
          }
          if (value.all === true) {
            return Object.freeze({ directory, sessions: "all" as const });
          }
          if (
            !Array.isArray(value.sessions) ||
            value.sessions.length === 0 ||
            value.sessions.length > 256 ||
            !value.sessions.every(
              (id): id is string =>
                typeof id === "string" &&
                id.length > 0 &&
                id.length <= 512 &&
                id.trim() === id &&
                !containsAsciiControlCharacter(id),
            )
          ) {
            throw new Error("invalid_control");
          }
          return Object.freeze({
            directory,
            sessions: Object.freeze([...new Set(value.sessions)]),
          });
        } finally {
          await file.close();
        }
      })(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error("control_timeout"));
        }, 10_000);
      }),
    ]);
  } catch (error) {
    if (!isRecord(error) || error.code !== "ENOENT") {
      // Never include the path, selector, exception text, or session identity.
      log.warn("capture_control_unavailable");
    }
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

/** Process-start admission, including failed/off results; never refresh on config reload. */
export function initializeAzureResponsesCaptureScope(cfg: OpenClawConfig): Promise<void> {
  startup ??= (async () => {
    const workspaces = new Map<string, Promise<Selection | undefined>>();
    const admitted = new Map<string, Selection>();
    await Promise.all(
      listAgentIds(cfg).map(async (agentId) => {
        const workspace = path.resolve(resolveAgentWorkspaceDir(cfg, agentId));
        let selection = workspaces.get(workspace);
        if (!selection) {
          selection = readSelection(workspace);
          workspaces.set(workspace, selection);
        }
        const value = await selection;
        if (value) {
          admitted.set(agentId, value);
        }
      }),
    );
    selections = admitted;
  })();
  return startup;
}

export function resolveAzureResponsesCaptureSelection(agentId: string, sessionId: string) {
  const selection = selections.get(agentId);
  return selection && (selection.sessions === "all" || selection.sessions.includes(sessionId))
    ? selection.directory
    : undefined;
}
