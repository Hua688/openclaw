import { chmod, mkdir, rename, symlink, unlink, writeFile } from "node:fs/promises";
import * as fs from "node:fs/promises";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useAutoCleanupTempDirTracker } from "../../test/helpers/temp-dir.js";

const warn = vi.hoisted(() => vi.fn());
vi.mock("../logging/subsystem.js", () => ({ createSubsystemLogger: () => ({ warn }) }));
vi.mock("node:fs/promises", async (original) => {
  const actual = await original<typeof import("node:fs/promises")>();
  return { ...actual, open: vi.fn(actual.open) };
});
const tempDirs = useAutoCleanupTempDirTracker(afterEach);
afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(fs.open).mockClear();
  warn.mockClear();
});

async function prepareWorkspace(control?: unknown) {
  const workspace = tempDirs.make("capture-boot-");
  const directory = path.join(workspace, ".openclaw", "azure-responses-cache-capture");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await chmod(directory, 0o700);
  const file = path.join(directory, ".enabled");
  if (control !== undefined) {
    await writeFile(file, JSON.stringify(control), { mode: 0o600 });
  }
  return { workspace, directory, file };
}
async function processOwner() {
  vi.resetModules();
  return import("./azure-responses-cache-capture-scope.js");
}

describe("process-start capture admission", () => {
  it("reads each canonical workspace once, freezes selected sessions and admits a new process's changes", async () => {
    const single = await prepareWorkspace({ sessions: ["single"] });
    const multiple = await prepareWorkspace({ sessions: ["one", "two"] });
    const all = await prepareWorkspace({ all: true });
    const off = await prepareWorkspace();
    const cfg = {
      agents: {
        ownership: "explicit" as const,
        entries: {
          single: { workspace: single.workspace },
          shared: { workspace: single.workspace },
          multiple: { workspace: multiple.workspace },
          all: { workspace: all.workspace },
          off: { workspace: off.workspace },
        },
      },
    };
    const owner = await processOwner();
    const open = vi.mocked(fs.open);
    const pending = owner.initializeAzureResponsesCaptureScope(cfg);
    expect(owner.initializeAzureResponsesCaptureScope(cfg)).toBe(pending);
    expect(owner.resolveAzureResponsesCaptureSelection("single", "single")).toBeUndefined();
    await pending;
    expect(open).toHaveBeenCalledTimes(3);
    expect(owner.resolveAzureResponsesCaptureSelection("single", "single")).toBe(single.directory);
    expect(owner.resolveAzureResponsesCaptureSelection("shared", "single")).toBe(single.directory);
    expect(owner.resolveAzureResponsesCaptureSelection("single", "other")).toBeUndefined();
    for (const id of ["one", "two"]) {
      expect(owner.resolveAzureResponsesCaptureSelection("multiple", id)).toBe(multiple.directory);
    }
    expect(owner.resolveAzureResponsesCaptureSelection("all", "cron-session")).toBe(all.directory);
    expect(owner.resolveAzureResponsesCaptureSelection("unknown", "cron-session")).toBeUndefined();
    await writeFile(single.file, '{"sessions":["changed"]}', { mode: 0o600 });
    await unlink(all.file);
    await writeFile(off.file, '{"all":true}', { mode: 0o600 });
    await owner.initializeAzureResponsesCaptureScope(cfg);
    expect(open).toHaveBeenCalledTimes(3);
    expect(owner.resolveAzureResponsesCaptureSelection("single", "single")).toBe(single.directory);
    expect(owner.resolveAzureResponsesCaptureSelection("single", "changed")).toBeUndefined();
    expect(owner.resolveAzureResponsesCaptureSelection("all", "cron-session")).toBe(all.directory);
    expect(owner.resolveAzureResponsesCaptureSelection("off", "any")).toBeUndefined();
    const restarted = await processOwner();
    await restarted.initializeAzureResponsesCaptureScope(cfg);
    expect(restarted.resolveAzureResponsesCaptureSelection("single", "single")).toBeUndefined();
    expect(restarted.resolveAzureResponsesCaptureSelection("single", "changed")).toBe(
      single.directory,
    );
    expect(restarted.resolveAzureResponsesCaptureSelection("all", "cron-session")).toBeUndefined();
    expect(restarted.resolveAzureResponsesCaptureSelection("off", "any")).toBe(off.directory);
  });

  it.each([
    "missing",
    "invalid",
    "permission",
    "io",
    "malformed",
    "unsafe-file",
    "unsafe-directory",
    "unsafe-parent",
    "oversized",
  ] as const)(
    "caches %s as off without leaking control contents or preventing startup",
    async (mode) => {
      const fixture = await prepareWorkspace(
        mode === "missing" ? undefined : { sessions: ["private-session"] },
      );
      if (mode === "invalid") {
        await writeFile(fixture.file, '{"all":true,"sessions":["private-session"]}');
      } else if (mode === "malformed") {
        await writeFile(fixture.file, "not-json private-session");
      } else if (mode === "unsafe-file") {
        const target = path.join(fixture.directory, "control-target");
        await rename(fixture.file, target);
        await symlink(target, fixture.file, "file");
      } else if (mode === "unsafe-directory") {
        const target = `${fixture.directory}.original`;
        await rename(fixture.directory, target);
        await symlink(target, fixture.directory, process.platform === "win32" ? "junction" : "dir");
      } else if (mode === "unsafe-parent") {
        const parent = path.dirname(fixture.directory);
        const target = `${parent}.original`;
        await rename(parent, target);
        await symlink(target, parent, process.platform === "win32" ? "junction" : "dir");
      } else if (mode === "oversized") {
        await writeFile(fixture.file, " ".repeat(65_537), { mode: 0o600 });
      }
      const owner = await processOwner();
      const open = vi.mocked(fs.open);
      if (mode === "permission" || mode === "io") {
        open.mockRejectedValueOnce(
          Object.assign(new Error("private-session private-path"), {
            code: mode === "permission" ? "EACCES" : "EIO",
          }),
        );
      }
      const cfg = { agents: { entries: { main: { workspace: fixture.workspace } } } };
      await expect(owner.initializeAzureResponsesCaptureScope(cfg)).resolves.toBeUndefined();
      expect(
        owner.resolveAzureResponsesCaptureSelection("main", "private-session"),
      ).toBeUndefined();
      const calls = open.mock.calls.length;
      await owner.initializeAzureResponsesCaptureScope(cfg);
      expect(open).toHaveBeenCalledTimes(calls);
      expect(warn.mock.calls).toEqual(mode === "missing" ? [] : [["capture_control_unavailable"]]);
    },
  );
});
