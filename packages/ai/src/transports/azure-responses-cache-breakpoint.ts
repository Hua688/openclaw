import { resolveOpenAIThinkingApi } from "@openclaw/model-catalog-core/model-catalog-types";
import type { ResponseInput } from "openai/resources/responses/responses.js";
import { resolveCacheRetention } from "../providers/cache-retention.js";
import { supportsOpenAIPromptCacheBreakpoints } from "../providers/openai-prompt-cache.js";
import type { Context, Model, StreamOptions } from "../types.js";
import {
  OPENAI_RESPONSES_COMPACTION_REPLAY_TYPE,
  OPENAI_RESPONSES_RETAINED_COMPACTION_REPLAY_TYPE,
  type OpenAIResponsesRequestParams,
} from "./openai-responses-contracts.js";
import { bindResponsesInputMessage } from "./openai-responses-replay-messages-internal.js";

type BreakpointRequest = Pick<OpenAIResponsesRequestParams, "model" | "input">;
type PreparedBreakpoint = (request: BreakpointRequest, model: Model) => ResponseInput | undefined;

const preparedBreakpoints = new WeakMap<BreakpointRequest, PreparedBreakpoint>();

/** Bind original user identities before conversion; never rebind live history objects. */
function prepareAzureResponsesCacheBreakpoint(
  model: Model,
  context: Context,
  options?: Pick<StreamOptions, "cacheRetention">,
) {
  if (
    resolveOpenAIThinkingApi(model.api) !== "azure-openai-responses" ||
    !supportsOpenAIPromptCacheBreakpoints({ id: model.id.toLowerCase() }) ||
    resolveCacheRetention(options?.cacheRetention) === "none" ||
    context.messages.some(
      (message) =>
        message.role === "assistant" &&
        (message.providerReplay?.type === OPENAI_RESPONSES_COMPACTION_REPLAY_TYPE ||
          message.providerReplay?.type === OPENAI_RESPONSES_RETAINED_COMPACTION_REPLAY_TYPE),
    ) ||
    context.messages.filter(
      (message) =>
        message.role === "user" &&
        message.runtimeContextCarrier === true &&
        message.runtimeContextCarrierRetained !== true,
    ).length !== 1
  ) {
    return undefined;
  }
  const route = {
    id: model.id,
    api: model.api,
    provider: model.provider,
    baseUrl: model.baseUrl,
  };
  const sources: Array<{
    matches: (input: unknown) => boolean;
    carrier: boolean;
    current: boolean;
  }> = [];
  const preparedContext: Context = {
    ...context,
    messages: context.messages.map((message) => {
      if (message.role !== "user") {
        return message;
      }
      const source = { ...message };
      sources.push({
        matches: bindResponsesInputMessage(source),
        carrier: message.runtimeContextCarrier === true,
        current:
          message.runtimeContextCarrier === true && message.runtimeContextCarrierRetained !== true,
      });
      return source;
    }),
  };
  return {
    context: preparedContext,
    bind(request: BreakpointRequest): void {
      if (!Array.isArray(request.input)) {
        return;
      }
      const deploymentName = request.model;
      const count = request.input.length;
      const carriers = new Map<number, string>();
      const texts = new Map<string, string>();
      let currentIndex: number | undefined;
      request.input.forEach((item, index) => {
        const source = sources.find(({ matches }) => matches(item));
        if (!source) {
          return;
        }
        if (source.carrier) {
          carriers.set(index, JSON.stringify(item));
          if (source.current) {
            currentIndex = index;
          }
        } else if ("content" in item && Array.isArray(item.content)) {
          item.content.forEach((block, blockIndex) => {
            if (block.type === "input_text") {
              texts.set(`${index}:${blockIndex}`, JSON.stringify(block));
            }
          });
        }
      });
      if (currentIndex !== count - 1) {
        return;
      }
      preparedBreakpoints.set(request, (payload, currentModel) => {
        const items = payload.input;
        if (
          payload.model !== deploymentName ||
          currentModel.id !== route.id ||
          currentModel.api !== route.api ||
          currentModel.provider !== route.provider ||
          currentModel.baseUrl !== route.baseUrl ||
          !Array.isArray(items) ||
          items.length !== count ||
          [...carriers].some(([index, bytes]) => JSON.stringify(items[index]) !== bytes) ||
          items.some(
            (item) =>
              "content" in item &&
              Array.isArray(item.content) &&
              item.content.some((block) => "prompt_cache_breakpoint" in block),
          )
        ) {
          return undefined;
        }
        // Original block bytes exclude callback-created text and image-sanitizer placeholders.
        for (let index = items.length - 2; index >= 0; index--) {
          const item = items[index];
          if (
            !item ||
            carriers.has(index) ||
            !("role" in item) ||
            item.role !== "user" ||
            ("type" in item && item.type !== "message") ||
            !("content" in item) ||
            !Array.isArray(item.content)
          ) {
            continue;
          }
          for (let blockIndex = item.content.length - 1; blockIndex >= 0; blockIndex--) {
            const block = item.content[blockIndex];
            if (
              !block ||
              block.type !== "input_text" ||
              texts.get(`${index}:${blockIndex}`) !== JSON.stringify(block)
            ) {
              continue;
            }
            const content = item.content.slice();
            content[blockIndex] = { ...block, prompt_cache_breakpoint: { mode: "explicit" } };
            const input = items.slice();
            input[index] = { ...item, content };
            return input;
          }
        }
        return undefined;
      });
    },
  };
}

/** Call-local prepared facts survive callback replacement and sanitizer cloning, not requests. */
function applyAzureResponsesCacheBreakpoint<T extends BreakpointRequest>(
  original: BreakpointRequest,
  request: T,
  model: Model,
): T {
  const prepared = preparedBreakpoints.get(original);
  preparedBreakpoints.delete(original);
  const input = prepared?.(request, model);
  return input ? { ...request, input } : request;
}

export const azureResponsesCacheBreakpoint = {
  build<T extends BreakpointRequest>(
    model: Model,
    context: Context,
    options: Pick<StreamOptions, "cacheRetention"> | undefined,
    buildRequest: (context: Context) => T,
    eligible = true,
  ): T {
    const prepared = eligible
      ? prepareAzureResponsesCacheBreakpoint(model, context, options)
      : undefined;
    const request = buildRequest(prepared?.context ?? context);
    prepared?.bind(request);
    return request;
  },
  apply: applyAzureResponsesCacheBreakpoint,
};
