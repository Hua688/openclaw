import { resolveOpenAIThinkingApi } from "@openclaw/model-catalog-core/model-catalog-types";
import { isEmbeddedMode } from "../../../infra/embedded-mode.js";
import {
  AZURE_RESPONSES_CAPTURE_CONTEXT,
  cacheCaptureDigest,
  getAzureResponsesCaptureScope,
  readSkillsSnapshotCaptureFacts,
  readToolPreparationCaptureFacts,
} from "../../azure-responses-cache-tracking.js";
import type { EmbeddedAttemptExecutionPhaseInput } from "./attempt-execution-types.js";

export function installAzureResponsesCapture(input: EmbeddedAttemptExecutionPhaseInput): void {
  const { attempt, prepared } = input;
  const { sessionRuntime, promptToolPolicy } = prepared;
  const { activeSession } = sessionRuntime.agentSession;
  const { appendOnlyRuntimeContext } = sessionRuntime.transcriptPolicy;
  const scope =
    resolveOpenAIThinkingApi(attempt.model.api) === "azure-openai-responses"
      ? getAzureResponsesCaptureScope(input.setup.sessionAgentId, attempt.sessionId)
      : undefined;
  if (!scope) {
    return;
  }
  const streamFn = activeSession.agent.streamFn;
  let requestIndex = 0;
  activeSession.agent.streamFn = (model, context, options) => {
    const isAzure = resolveOpenAIThinkingApi(model.api) === "azure-openai-responses";
    if (activeSession.isCompacting || !isAzure) {
      return streamFn(model, context, options);
    }
    const { skillsSnapshot: snapshot, trigger } = attempt;
    const policy = promptToolPolicy.current;
    const catalog = prepared.toolCatalog.toolSearch;
    const facts = {
      origin: {
        trigger: ["user", "heartbeat", "cron", "memory", "overflow", "manual"].includes(
          trigger ?? "",
        )
          ? trigger
          : "other",
        userRequestEvent: attempt.currentInboundEventKind === "user_request",
        roomEvent: attempt.currentInboundEventKind === "room_event",
        messageToolOnlyDelivery: attempt.sourceReplyDeliveryMode === "message_tool_only",
        sourceReplyDeliveryMode: attempt.sourceReplyDeliveryMode
          ? { status: "captured", value: attempt.sourceReplyDeliveryMode }
          : { status: "unavailable", reason: "source-reply-delivery-mode-not-present" },
        taskSuggestionDeliveryMode: attempt.taskSuggestionDeliveryMode
          ? { status: "captured", value: attempt.taskSuggestionDeliveryMode }
          : { status: "unavailable", reason: "task-suggestion-delivery-mode-not-present" },
        sessionKeyPresent: Boolean(attempt.sessionKey),
        embedded: isEmbeddedMode(),
      },
      skillsSnapshot: {
        producer: readSkillsSnapshotCaptureFacts(snapshot, attempt.sessionId),
        attempt: snapshot
          ? {
              status: "captured",
              version: snapshot.version,
              prompt: {
                status: "captured",
                charCount: snapshot.prompt.length,
                sha256: cacheCaptureDigest(snapshot.prompt),
              },
              skillCount: snapshot.skills.length,
              resolvedSkillCount: snapshot.resolvedSkills?.length,
            }
          : { status: "unavailable", reason: "attempt-snapshot-not-present" },
      },
      tools: {
        ownerStages: readToolPreparationCaptureFacts(attempt),
        catalog: {
          status: "captured",
          registered: catalog.catalogRegistered,
          reused: catalog.catalogReused,
          catalogToolCount: catalog.catalogToolCount,
        },
        promptPolicy: {
          status: "captured",
          activeToolCount: policy.activeToolNames.length,
          activeToolNamesSha256: cacheCaptureDigest(JSON.stringify(policy.activeToolNames)),
          callableToolCount: policy.callableToolNames.length,
          callableToolNamesSha256: cacheCaptureDigest(JSON.stringify(policy.callableToolNames)),
          effectiveToolCount: policy.effectiveTools.length,
          effectiveToolNamesSha256: cacheCaptureDigest(
            JSON.stringify(policy.effectiveTools.map((tool) => tool.name)),
          ),
          targetTools: {
            screen: policy.activeToolNames.includes("screen"),
            suggestTask: policy.activeToolNames.includes("suggest_task"),
            dismissTask: policy.activeToolNames.includes("dismiss_task"),
          },
        },
      },
      runtimeContext: {
        appendOnlyPolicy: { status: "captured", value: appendOnlyRuntimeContext === true },
        normalization: sessionRuntime.boundary.getCacheTrackingFacts?.() ?? {
          status: "unavailable",
          reason: "wire-conversion-facts-not-observed",
        },
      },
    };
    return streamFn(
      model,
      context,
      Object.assign({}, options, {
        [AZURE_RESPONSES_CAPTURE_CONTEXT]: {
          ...scope,
          runHash: cacheCaptureDigest(attempt.runId),
          requestIndex: ++requestIndex,
          cacheTrackingFacts: () => facts,
        },
      }),
    );
  };
}
