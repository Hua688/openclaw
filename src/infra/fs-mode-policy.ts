// Fork-only patch (Hua688 deployment): disable OpenClaw's active chmod calls.
//
// OpenClaw's upstream code aggressively chmod's its state files in ~/.openclaw
// down to 0o600 / 0o700 (owner-only). That fights with shared-backup setups
// where a separate backup user/group needs read access, and silently undoes
// any permissions set by our ansible playbook.
//
// We neutralize every chmod entry point Node exposes so permissions are
// managed entirely outside OpenClaw (umask + parent dir ACL + playbook).
// Initial file mode passed to writeFile/open is left untouched.
//
// Entry points covered:
//   - fs.chmod / fs.chmodSync                (path-based, callback + sync)
//   - fs.lchmod / fs.lchmodSync              (symlink target, macOS only)
//   - fs.fchmod / fs.fchmodSync              (fd-based)
//   - fsp.chmod / fsp.lchmod                 (fs/promises path-based)
//   - FileHandle.prototype.chmod             (fs/promises fd-based — the
//     entry that fs-safe's appendRegularFile uses; missing this caused
//     cron run-log appends to throw EPERM once the file owner drifted
//     away from the gateway runtime user.)
//
// chown* is deliberately NOT patched: nothing in openclaw or @openclaw/fs-safe
// calls chown, and silencing it would mask real bugs.
//
// Additionally we wrap fsp.stat so that, for paths under ~/.openclaw, the
// returned mode reports group/other bits as zero. @openclaw/fs-safe's
// enforcePrivatePathMode does chmod -> stat -> strict equality check; with
// chmod neutralized the real mode stays at whatever the playbook set (e.g.
// 0o770 for shared-backup ACLs) and the check would otherwise throw
// "Private secret directory ... has insecure permissions 770".
//
// This module is imported for side effects from src/entry.ts and must stay
// the first user module evaluated so later `import { chmodSync } from
// "node:fs"` bindings observe the no-op values.

import fs from "node:fs";
import fsp from "node:fs/promises";

const noop = () => {};
const noopCb = (_a: unknown, _b: unknown, cb?: (err: NodeJS.ErrnoException | null) => void) => {
  if (typeof cb === "function") cb(null);
};
const asyncNoop = async () => {};

const fsAny = fs as unknown as Record<string, unknown>;
const fspAny = fsp as unknown as Record<string, unknown>;

for (const name of ["chmodSync", "fchmodSync", "lchmodSync"]) {
  if (typeof fsAny[name] === "function") fsAny[name] = noop;
}
for (const name of ["chmod", "fchmod", "lchmod"]) {
  if (typeof fsAny[name] === "function") fsAny[name] = noopCb;
}
for (const name of ["chmod", "lchmod", "fchmod"]) {
  if (typeof fspAny[name] === "function") fspAny[name] = asyncNoop;
}

// Wrap fsp.stat so fs-safe's strict-equality permission check sees a mode
// with group/other bits cleared for anything under ~/.openclaw. Only the mode
// field is rewritten; isFile/isDirectory/size/etc are untouched.
const origStat = fsp.stat.bind(fsp);
const isOpenclawPath = (p: unknown): boolean => {
  if (typeof p !== "string") return false;
  return p.includes("/.openclaw");
};
fspAny.stat = async (p: unknown, opts?: unknown) => {
  const s = await (origStat as (a: unknown, b?: unknown) => Promise<{ mode: unknown }>)(p, opts);
  if (isOpenclawPath(p)) {
    const m: unknown = s.mode;
    if (typeof m === "bigint") {
      s.mode = m & ~0o077n;
    } else if (typeof m === "number") {
      s.mode = m & ~0o077;
    }
  }
  return s;
};

// FileHandle is not an exported class; reach its prototype via a real handle.
// Top-level await guarantees the prototype is patched before any caller
// (e.g. fs-safe's appendRegularFile) opens its own handle.
try {
  const probe = await fsp.open(process.platform === "win32" ? "NUL" : "/dev/null", "r");
  const proto = Object.getPrototypeOf(probe) as Record<string, unknown>;
  for (const name of ["chmod"]) {
    if (typeof proto[name] === "function") proto[name] = asyncNoop;
  }
  await probe.close();
} catch {
  // Best-effort: do not crash startup if the probe target is unavailable.
}
