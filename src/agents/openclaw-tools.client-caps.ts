import { GATEWAY_CLIENT_CAPS } from "../../packages/gateway-protocol/src/client-info.js";
import type { AnyAgentTool } from "./tools/common.js";

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
 * Drops tools whose requiredClientCaps the originating gateway client did not
 * declare. Capability availability is a hard fact, not policy: every tool
 * assembly path (core, plugin-only plans) must apply it or gated tools leak
 * onto surfaces that cannot render them.
 */
export function filterToolsByClientCaps(
  tools: AnyAgentTool[],
  declaredClientCaps: string[] | undefined,
): AnyAgentTool[] {
  const clientCaps = new Set(declaredClientCaps ?? []);
  return tools.filter(
    (tool) => !tool.requiredClientCaps?.some((requiredCap) => !clientCaps.has(requiredCap)),
  );
}
