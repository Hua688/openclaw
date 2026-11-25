import { GATEWAY_CLIENT_CAPS } from "../../packages/gateway-protocol/src/client-info.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { gateAgentToolExecution } from "./agent-tool-availability.js";
import { resolveSessionCapabilityEntry } from "./subagents/spawn/subagent-capabilities.js";
import type {
  PreparedSessionCapabilityEntry,
  SessionCapabilityStore,
} from "./subagents/spawn/subagent-session-store.js";
import type { AnyAgentTool } from "./tools/common.js";

/** Read the same session-owned fact for coding and standalone tool assembly. */
export function resolveSessionToolListingClientCaps(params: {
  config?: OpenClawConfig;
  sessionKey?: string;
  runSessionKey?: string;
  agentSessionKey?: string;
  clientCaps?: string[];
  preparedSessionEntry?: PreparedSessionCapabilityEntry;
  preparedSessionCapabilityStore?: SessionCapabilityStore;
}): string[] {
  const sessionKey = params.runSessionKey ?? params.sessionKey ?? params.agentSessionKey;
  const entry = sessionKey
    ? params.preparedSessionEntry?.sessionKey === sessionKey
      ? params.preparedSessionEntry.entry
      : resolveSessionCapabilityEntry({
          sessionKey,
          cfg: params.config,
          store: params.preparedSessionCapabilityStore,
        })
    : undefined;
  return entry?.clientCapabilities ?? params.clientCaps ?? [];
}

export const CACHE_TRACKED_TOOL_NAMES = ["screen", "suggest_task", "dismiss_task"] as const;
export type ToolPreparationFacts =
  | {
      taskSuggestionEligibility: {
        embedded: boolean;
        sessionKeyPresent: boolean;
        taskSuggestionDeliveryMode: string;
        eligible: boolean;
      };
      trackedTools: Record<string, boolean>;
    }
  | {
      declaredCapsPresent: boolean;
      uiCommandsDeclared: boolean;
      tools: Record<
        string,
        {
          registered: boolean;
          registeredCount: number;
          clientCapEligible: boolean;
          requiredClientCapCount: number;
          includedAfterClientCaps: boolean;
          includedAfterAvailability: boolean;
        }
      >;
    };
export type ToolPreparationStageRecorder = (name: string, facts?: ToolPreparationFacts) => void;

export function summarizeCacheTrackedCoreTools(
  tools: readonly AnyAgentTool[],
  embedded: boolean,
  sessionKey: string | undefined,
  options: { taskSuggestionDeliveryMode?: string } | undefined,
): ToolPreparationFacts {
  return {
    taskSuggestionEligibility: {
      embedded,
      sessionKeyPresent: Boolean(sessionKey),
      taskSuggestionDeliveryMode: options?.taskSuggestionDeliveryMode ?? "unset",
      eligible:
        !embedded && Boolean(sessionKey) && options?.taskSuggestionDeliveryMode === "gateway",
    },
    trackedTools: Object.fromEntries(
      CACHE_TRACKED_TOOL_NAMES.map((name) => [name, tools.some((tool) => tool.name === name)]),
    ),
  };
}

export function summarizeCacheTrackedClientCaps(
  tools: readonly AnyAgentTool[],
  declaredClientCaps: string[] | undefined,
  filtered: readonly AnyAgentTool[],
  available: readonly AnyAgentTool[],
): ToolPreparationFacts {
  const clientCaps = new Set(declaredClientCaps ?? []);
  return {
    declaredCapsPresent: Array.isArray(declaredClientCaps),
    uiCommandsDeclared: clientCaps.has(GATEWAY_CLIENT_CAPS.UI_COMMANDS),
    tools: Object.fromEntries(
      CACHE_TRACKED_TOOL_NAMES.map((name) => {
        const matches = tools.filter((tool) => tool.name === name);
        return [
          name,
          {
            registered: matches.length > 0,
            registeredCount: matches.length,
            clientCapEligible: filtered.some((tool) => tool.name === name),
            requiredClientCapCount: matches.reduce(
              (count, tool) => count + (tool.requiredClientCaps?.length ?? 0),
              0,
            ),
            includedAfterClientCaps: filtered.some((tool) => tool.name === name),
            includedAfterAvailability: available.some((tool) => tool.name === name),
          },
        ];
      }),
    ),
  };
}

/**
 * List tools from the session's recorded operator capabilities, so internal
 * wakes retain the same catalog. Current-turn capabilities gate execution;
 * attached-client tools also check their live recipient at the point of effect.
 */
export function filterToolsByClientCaps(
  tools: AnyAgentTool[],
  listingClientCaps: string[] | undefined,
  executionClientCaps: string[] | undefined,
): AnyAgentTool[] {
  const listed = new Set(listingClientCaps ?? []);
  const current = new Set(executionClientCaps ?? []);
  return tools
    .filter((tool) => !tool.requiredClientCaps?.some((cap) => !listed.has(cap)))
    .map((tool) =>
      tool.requiredClientCaps?.some((cap) => !current.has(cap))
        ? gateAgentToolExecution(
            tool,
            `${tool.name} requires a connected requesting client with ${tool.requiredClientCaps.join(", ")}; ask again from the open Control UI.`,
          )
        : tool,
    );
}
