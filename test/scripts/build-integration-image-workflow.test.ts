import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parse } from "yaml";
import { useAutoCleanupTempDirTracker } from "../helpers/temp-dir.js";

type Workflow = {
  jobs: Record<string, { steps: Array<{ name: string; run?: string }> }>;
};

const workflow = parse(
  readFileSync(".github/workflows/build-integration-image.yml", "utf8"),
) as Workflow;
const tempDirs = useAutoCleanupTempDirTracker(afterEach);

function step(job: string, name: string): string {
  const script = workflow.jobs[job]?.steps.find((entry) => entry.name === name)?.run;
  if (!script) {
    throw new Error(`Missing workflow step: ${job}/${name}`);
  }
  return script;
}

function run(script: string, setup: string, env: Record<string, string>) {
  return spawnSync("/bin/bash", ["--noprofile", "--norc", "-c", `${setup}\n${script}`], {
    encoding: "utf8",
    timeout: 5_000,
    env: { PATH: "/usr/bin:/bin", ...env },
  });
}

describe.skipIf(process.platform === "win32")("integration image workflow", () => {
  it("dispatches the latest supported tag with scheduled skip enabled", () => {
    const root = tempDirs.make("integration-dispatch-");
    const args = join(root, "dispatch");
    const result = run(
      step("resolve-latest", "Resolve latest upstream tag and dispatch build"),
      `
git() { printf '%s\\n' "$TAGS"; }
gh() { printf '%s\\n' "$@" > "$ARGS"; }
`,
      {
        ARGS: args,
        GITHUB_REPOSITORY: "owner/repo",
        GITHUB_REF_NAME: "main",
        UPSTREAM_REPO: "https://example.invalid/repo",
        TAGS: [
          "sha\trefs/tags/v2026.11.3-beta.1",
          "sha\trefs/tags/v2026.11.3",
          "sha\trefs/tags/v2026.11.4-beta.2",
          "sha\trefs/tags/v2026.11.4-beta.10",
          "sha\trefs/tags/v2026.11.4-rc.1",
        ].join("\n"),
      },
    );
    expect(result.status).toBe(0);
    expect(readFileSync(args, "utf8").split("\n")).toEqual([
      "api",
      "--method",
      "POST",
      "repos/owner/repo/actions/workflows/build-integration-image.yml/dispatches",
      "-f",
      "ref=main",
      "-f",
      "inputs[upstream_ref]=v2026.11.4-beta.10",
      "-F",
      "inputs[skip_if_unchanged]=true",
      "",
    ]);
  });

  it("fails instead of dispatching when upstream has no supported tag", () => {
    const result = run(
      step("resolve-latest", "Resolve latest upstream tag and dispatch build"),
      `git() { printf '%s\\n' 'sha	refs/tags/not-a-release'; }
gh() { echo 'unexpected dispatch'; }`,
      { UPSTREAM_REPO: "https://example.invalid/repo" },
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("No supported upstream release tag found");
    expect(result.stdout).not.toContain("unexpected dispatch");
  });

  it("reports a failed dispatch as a failed resolver", () => {
    const result = run(
      step("resolve-latest", "Resolve latest upstream tag and dispatch build"),
      `git() { printf '%s\\n' 'sha	refs/tags/v2026.11.3'; }
gh() { return 42; }`,
      {
        UPSTREAM_REPO: "https://example.invalid/repo",
        GITHUB_REPOSITORY: "owner/repo",
        GITHUB_REF_NAME: "main",
      },
    );
    expect(result.status).toBe(42);
  });

  it.each([
    {
      scheduled: "true",
      patch: "patchsha",
      target: "v2026.11.3",
      previous: "v2026.11.3",
      expected: "true",
    },
    {
      scheduled: "false",
      patch: "patchsha",
      target: "v2026.11.3",
      previous: "v2026.11.3",
      expected: "false",
    },
    {
      scheduled: "true",
      patch: "newpatchsha",
      target: "v2026.11.3",
      previous: "v2026.11.3",
      expected: "false",
    },
    {
      scheduled: "true",
      patch: "patchsha",
      target: "v2026.11.3",
      previous: "v2026.11.4",
      expected: "stale",
    },
    {
      scheduled: "false",
      patch: "patchsha",
      target: "v2026.11.3",
      previous: "v2026.11.4",
      expected: "false",
    },
    {
      scheduled: "true",
      patch: "patchsha",
      target: "v2026.11.3-beta.2",
      previous: "v2026.11.3",
      expected: "stale",
    },
    {
      scheduled: "true",
      patch: "patchsha",
      target: "v2026.11.3-beta.2",
      previous: "v2026.11.3-beta.10",
      expected: "stale",
    },
  ])("handles $scheduled/$target after $previous with $patch", (test) => {
    const root = tempDirs.make("integration-detect-");
    const output = join(root, "output");
    const source = "patch-a patchsha v2026.11.1 basesha 1\n";
    const fingerprint = createHash("sha256").update(source).digest("hex");
    const result = run(
      step("build", "Detect upstream tag and snapshot patch inputs"),
      `
git() {
  case "$1 $2" in
    "rev-parse --verify")
      case "$3" in
        refs/tags/v2026.11.3*) echo targetsha ;;
        refs/tags/v2026.11.1*) echo basesha ;;
        refs/remotes/origin/integration/Hua688*) echo previoussha ;;
        refs/remotes/origin/patch-a*) echo "$PATCH_HEAD" ;;
        *) return 1 ;;
      esac ;;
    "ls-remote --heads") printf 'previoussha\\trefs/heads/integration/Hua688\\n' ;;
    "fetch origin") : ;;
    "show -s") printf '%s\\n' "$METADATA" ;;
    "rev-list --first-parent") printf '%s\\n' "$PATCH_HEAD" basesha ;;
    "rev-list --count") echo 1 ;;
    "describe --first-parent") echo v2026.11.1 ;;
    *) printf 'unexpected git invocation: %s\\n' "$*" >&2; return 1 ;;
  esac
}
`,
      {
        UPSTREAM_REF: test.target,
        SKIP_IF_UNCHANGED: test.scheduled,
        PATCH_HEAD: test.patch,
        PATCHES: "patch-a",
        GITHUB_OUTPUT: output,
        GITHUB_STEP_SUMMARY: join(root, "summary"),
        METADATA: [
          `OpenClaw-Integration-Target: ${test.previous}`,
          "OpenClaw-Integration-Target-SHA: targetsha",
          `OpenClaw-Patch-Fingerprint: ${fingerprint}`,
        ].join("\n"),
      },
    );
    if (test.expected === "stale") {
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("refusing to overwrite");
    } else {
      expect(result.stderr).toBe("");
      expect(result.status).toBe(0);
      expect(readFileSync(output, "utf8")).toContain(`skip=${test.expected}\n`);
    }
  });
});
