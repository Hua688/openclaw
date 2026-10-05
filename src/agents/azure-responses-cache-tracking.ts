import { createHash } from "node:crypto";
import path from "node:path";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { resolveRuntimeContextPromptOwner } from "./internal-runtime-context.js";
import type { ToolPreparationFacts } from "./openclaw-tools.client-caps.js";

export const AZURE_RESPONSES_CAPTURE_CONTEXT = Symbol.for(
  "openclaw.azureResponsesCacheCaptureContext.v1",
);
const SKILLS_FACTS = Symbol.for("openclaw.azureResponsesCacheTrackingFacts.v1.skillsSnapshot");
const TOOL_FACTS = Symbol.for("openclaw.azureResponsesCacheTrackingFacts.v1.toolPreparation");
const log = createSubsystemLogger("responses-cache-capture");

export function cacheCaptureDigest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

const CAPTURE_SCOPE = Symbol.for("openclaw.azureResponsesCacheCaptureScope.v1");

/** Trusted same-process diagnostic scope; absent by default and not product configuration. */
export function getAzureResponsesCaptureScope(sessionId: string) {
  const scope: unknown = Reflect.get(globalThis, CAPTURE_SCOPE);
  if (scope === undefined) {
    return undefined;
  }
  const directory = isRecord(scope) ? scope.directory : undefined;
  const sessionHash = isRecord(scope) ? scope.sessionHash : undefined;
  if (
    typeof directory !== "string" ||
    !path.isAbsolute(directory) ||
    typeof sessionHash !== "string" ||
    !/^[a-f\d]{64}$/i.test(sessionHash)
  ) {
    log.warn("invalid_scope");
    return undefined;
  }
  if (cacheCaptureDigest(sessionId) !== sessionHash.toLowerCase()) {
    return undefined;
  }
  return {
    sessionHash: sessionHash.toLowerCase(),
    directory: path.resolve(directory),
    enabledFile: path.join(path.resolve(directory), ".enabled"),
    minimumFreeBytes: 1024 * 1024 * 1024,
  };
}

export type SkillsSnapshotRuntimeFacts = {
  initial: { shouldRefresh: boolean; snapshotVersion: number };
  latestResolution?: { shouldRefresh: boolean; snapshotVersion: number };
  persistence: "not-attempted" | "updated" | "stale";
  selectedSnapshotVersion?: number;
  persistedSnapshotVersion?: number;
};

export function recordSkillsSnapshotCaptureFacts(
  snapshot: object | undefined,
  sessionId: string | undefined,
  facts: SkillsSnapshotRuntimeFacts,
): void {
  if (snapshot && sessionId && getAzureResponsesCaptureScope(sessionId)) {
    const recorded = Reflect.defineProperty(snapshot, SKILLS_FACTS, {
      value: { sessionId, facts },
      enumerable: true,
      configurable: true,
      writable: true,
    });
    if (!recorded) {
      log.warn("skills_snapshot_facts_not_recorded");
    }
  }
}

export function readSkillsSnapshotCaptureFacts(snapshot: unknown, sessionId: string) {
  const carrier: unknown =
    snapshot && typeof snapshot === "object" ? Reflect.get(snapshot, SKILLS_FACTS) : undefined;
  return isRecord(carrier) && carrier.sessionId === sessionId
    ? { status: "captured", facts: carrier.facts }
    : {
        status: "unavailable",
        reason: carrier
          ? "snapshot-producer-session-mismatch"
          : "snapshot-producer-facts-not-carried",
      };
}

export function recordToolPreparationCaptureFacts(
  attempt: object,
  stage: string,
  facts: ToolPreparationFacts | undefined,
) {
  if (
    !facts ||
    !["openclaw-tools:core-tool-list", "openclaw-tools:client-capabilities"].includes(stage)
  ) {
    return;
  }
  const current: unknown = Reflect.get(attempt, TOOL_FACTS);
  const stages: unknown[] = Array.isArray(current)
    ? current.filter((item) => !isRecord(item) || item.stage !== stage)
    : [];
  stages.push({ stage, facts });
  if (
    !Reflect.defineProperty(attempt, TOOL_FACTS, {
      value: stages,
      enumerable: false,
      configurable: true,
    })
  ) {
    log.warn("tool_preparation_facts_not_recorded");
  }
}

export function readToolPreparationCaptureFacts(attempt: object) {
  const facts: unknown = Reflect.get(attempt, TOOL_FACTS);
  return Array.isArray(facts)
    ? { status: "captured", stages: facts.slice() }
    : { status: "unavailable", reason: "tool-owner-stages-not-carried" };
}

export function startRuntimeContextNormalizationCapture(
  messages: readonly unknown[],
  appendOnlyPolicy: boolean | undefined,
  retainedOwnerPresent: boolean,
) {
  const input = summarizeRuntimeContextCarriers(messages);
  const stages: Partial<
    Record<"normalized" | "positioned", ReturnType<typeof summarizeRuntimeContextCarriers>>
  > = {};
  return {
    record(stage: "normalized" | "positioned", stageMessages: readonly unknown[]) {
      stages[stage] = summarizeRuntimeContextCarriers(stageMessages);
    },
    finish(converted: readonly unknown[], currentPlacementEligible: boolean) {
      const { normalized, positioned } = stages;
      if (!normalized || !positioned) {
        return { status: "unavailable", reason: "normalization-stage-not-observed" };
      }
      return {
        status: "captured",
        appendOnlyPolicy: appendOnlyPolicy === true,
        retainedOwnerPresent,
        currentPlacementEligible,
        currentPlacementApplied: JSON.stringify(normalized) !== JSON.stringify(positioned),
        historicalCarrierRemovedCount: Math.max(0, input.count - normalized.count),
        stages: {
          input,
          normalized,
          positioned,
          converted: summarizeRuntimeContextCarriers(converted),
        },
      };
    },
  };
}

export function summarizeRuntimeContextCarriers(messages: readonly unknown[]) {
  const carriers = messages.flatMap((message, index) => {
    if (!isRecord(message)) {
      return [];
    }

    const details = message.details;
    const user = message.role === "user" && message.runtimeContextCarrier === true;
    const custom =
      message.role === "custom" &&
      (message.customType === "openclaw.runtime-context" ||
        (isRecord(details) &&
          details.source === "openclaw-runtime-context" &&
          details.runtimeContextCarrier === true));
    if (!user && !custom) {
      return [];
    }
    const text =
      typeof message.content === "string"
        ? message.content
        : Array.isArray(message.content)
          ? message.content
              .flatMap((part) =>
                isRecord(part) && typeof part.text === "string" ? [part.text] : [],
              )
              .join("\n")
          : undefined;
    return [
      {
        index,
        kind: user ? "user-carrier" : "custom-carrier",
        retention: !user
          ? { status: "not-applicable" }
          : typeof message.runtimeContextCarrierRetained === "boolean"
            ? { status: "captured", retained: message.runtimeContextCarrierRetained }
            : { status: "unavailable", reason: "retention-flag-not-materialized" },
        ...(text !== undefined
          ? { contentChars: text.length, contentSha256: cacheCaptureDigest(text) }
          : {}),
      },
    ];
  });
  const owner = resolveRuntimeContextPromptOwner(messages);
  return {
    status: "captured",
    count: carriers.length,
    promptOwner: owner ? { status: "retained", userIndex: owner.userIndex } : { status: "absent" },
    carriers,
  };
}
