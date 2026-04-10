export function isTraditionalAzureOpenAIHost(hostname: string): boolean {
  return (
    hostname.endsWith(".openai.azure.com") || hostname.endsWith(".cognitiveservices.azure.com")
  );
}

export function isOpenAICompatibleAzureResponsesBaseUrl(baseUrl: string): boolean {
  const url = URL.parse(baseUrl);
  if (!url || isTraditionalAzureOpenAIHost(url.hostname)) {
    return false;
  }

  const hostname = url.hostname.toLowerCase();
  const isFoundryHost =
    hostname.endsWith(".services.ai.azure.com") ||
    hostname.endsWith(".api.cognitive.microsoft.com");
  if (!isFoundryHost) {
    return false;
  }

  const normalizedPath = url.pathname.replace(/\/+$/, "");
  return normalizedPath === "/openai/v1" || normalizedPath.endsWith("/openai/v1");
}

export function resolveAzureOpenAIBaseUrl(params: {
  modelBaseUrl?: string;
  azureBaseUrl?: string;
  azureResourceName?: string;
}): string {
  const baseUrl =
    params.azureBaseUrl?.trim() || process.env.AZURE_OPENAI_BASE_URL?.trim() || undefined;
  const resourceName = params.azureResourceName || process.env.AZURE_OPENAI_RESOURCE_NAME;
  const resolvedBaseUrl =
    baseUrl ||
    (resourceName ? `https://${resourceName}.openai.azure.com/openai/v1` : params.modelBaseUrl);

  if (!resolvedBaseUrl) {
    throw new Error(
      "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.",
    );
  }

  const trimmed = resolvedBaseUrl.trim().replace(/\/+$/, "");
  const url = URL.parse(trimmed);
  if (!url) {
    throw new Error(`Invalid Azure OpenAI base URL: ${resolvedBaseUrl}`);
  }

  const normalizedPath = url.pathname.replace(/\/+$/, "");
  if (
    isTraditionalAzureOpenAIHost(url.hostname) &&
    (normalizedPath === "" || normalizedPath === "/" || normalizedPath === "/openai")
  ) {
    url.pathname = "/openai/v1";
    url.search = "";
  }

  return url.toString().replace(/\/+$/, "");
}
