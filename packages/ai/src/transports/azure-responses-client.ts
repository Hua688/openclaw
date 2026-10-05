import type { Model } from "@openclaw/llm-core";
import OpenAI, { AzureOpenAI } from "openai";
import { isOpenAICompatibleAzureResponsesBaseUrl } from "../providers/azure-openai-responses-client-compat.js";
import { buildGuardedModelFetch } from "./host-policy.js";
import { resolveAzureOpenAIApiVersion } from "./openai-responses-replay-internal.js";
import { buildOpenAISdkClientOptions } from "./openai-transport-params.js";

export function createAzureOpenAIClient(
  model: Model,
  apiKey: string,
  defaultHeaders: Record<string, string>,
  fetchOverride?: typeof globalThis.fetch,
) {
  const baseURL = model.baseUrl.replace(/\/+$/, "");
  const clientOptions = {
    apiKey,
    dangerouslyAllowBrowser: true,
    defaultHeaders,
    baseURL,
    fetch: fetchOverride ?? buildGuardedModelFetch(model),
    ...buildOpenAISdkClientOptions(model),
  };

  if (isOpenAICompatibleAzureResponsesBaseUrl(baseURL)) {
    return new OpenAI(clientOptions);
  }

  return new AzureOpenAI({
    ...clientOptions,
    apiVersion: resolveAzureOpenAIApiVersion(),
  });
}
