import { Type } from "@sinclair/typebox";
import {
  buildSearchCacheKey,
  DEFAULT_SEARCH_COUNT,
  MAX_SEARCH_COUNT,
  readCachedSearchPayload,
  readConfiguredSecretString,
  readNumberParam,
  readProviderEnvValue,
  readStringParam,
  resolveProviderWebSearchPluginConfig,
  resolveSearchCacheTtlMs,
  resolveSearchCount,
  resolveSearchTimeoutSeconds,
  resolveSiteName,
  setProviderWebSearchPluginConfigValue,
  type SearchConfigRecord,
  type WebSearchProviderPlugin,
  type WebSearchProviderToolDefinition,
  withTrustedWebSearchEndpoint,
  wrapWebContent,
  writeCachedSearchPayload,
} from "openclaw/plugin-sdk/provider-web-search";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SearxngResult = {
  title?: string;
  url?: string;
  content?: string;
  engine?: string;
  publishedDate?: string;
};

type SearxngResponse = {
  results?: SearxngResult[];
  number_of_results?: number;
};

type SearxngPluginConfig = {
  baseUrl?: string;
  apiKey?: unknown;
};

// ---------------------------------------------------------------------------
// Config resolution
// ---------------------------------------------------------------------------

function resolvePluginConfig(
  searchConfig: SearchConfigRecord | undefined,
  config: Record<string, unknown> | undefined,
): SearxngPluginConfig {
  const pluginConf = resolveProviderWebSearchPluginConfig(config, "searxng");
  const webSearch =
    pluginConf && typeof pluginConf === "object" ? (pluginConf as SearxngPluginConfig) : {};

  // Also check legacy top-level searxng config in search block
  const legacy =
    searchConfig && typeof searchConfig === "object"
      ? (searchConfig as Record<string, unknown>)
      : undefined;
  const legacySearxng =
    legacy?.searxng && typeof legacy.searxng === "object"
      ? (legacy.searxng as SearxngPluginConfig)
      : undefined;

  return {
    baseUrl: webSearch.baseUrl ?? legacySearxng?.baseUrl,
    apiKey: webSearch.apiKey ?? legacySearxng?.apiKey,
  };
}

function resolveSearxngBaseUrl(pluginConfig: SearxngPluginConfig): string | undefined {
  const fromConfig = typeof pluginConfig.baseUrl === "string" ? pluginConfig.baseUrl.trim() : "";
  const fromEnv = readProviderEnvValue(["SEARXNG_BASE_URL"]);
  const raw = fromConfig || fromEnv || undefined;
  return raw ? raw.replace(/\/+$/, "") : undefined;
}

function resolveSearxngApiKey(pluginConfig: SearxngPluginConfig): string | undefined {
  return (
    readConfiguredSecretString(
      pluginConfig.apiKey,
      "plugins.entries.searxng.config.webSearch.apiKey",
    ) ?? readProviderEnvValue(["SEARXNG_API_KEY"])
  );
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

function createSearxngSchema() {
  return Type.Object({
    query: Type.String({ description: "Search query." }),
    count: Type.Optional(
      Type.Number({
        description: "Number of results to return (1-10).",
        minimum: 1,
        maximum: MAX_SEARCH_COUNT,
      }),
    ),
    language: Type.Optional(
      Type.String({
        description: "Search language code (e.g., 'en', 'zh', 'de', 'all'). Default: auto-detect.",
      }),
    ),
    categories: Type.Optional(
      Type.String({
        description:
          "Comma-separated categories (e.g., 'general', 'news', 'images', 'videos', 'science'). Default: 'general'.",
      }),
    ),
  });
}

// ---------------------------------------------------------------------------
// Search execution
// ---------------------------------------------------------------------------

async function runSearxngSearch(params: {
  query: string;
  baseUrl: string;
  apiKey?: string;
  count: number;
  timeoutSeconds: number;
  language?: string;
  categories?: string;
}): Promise<Array<Record<string, unknown>>> {
  const url = new URL(`${params.baseUrl}/search`);
  url.searchParams.set("q", params.query);
  url.searchParams.set("format", "json");
  if (params.language) {
    url.searchParams.set("language", params.language);
  }
  if (params.categories) {
    url.searchParams.set("categories", params.categories);
  }
  url.searchParams.set("pageno", "1");

  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (params.apiKey) {
    headers.Authorization = `Bearer ${params.apiKey}`;
  }

  return withTrustedWebSearchEndpoint(
    {
      url: url.toString(),
      timeoutSeconds: params.timeoutSeconds,
      init: { method: "GET", headers },
    },
    async (res) => {
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`SearXNG API error (${res.status}): ${detail || res.statusText}`);
      }

      const data = (await res.json()) as SearxngResponse;
      const results = Array.isArray(data.results) ? data.results : [];

      // Deduplicate by URL (SearXNG may return the same URL from multiple engines)
      const seen = new Set<string>();
      const deduped: SearxngResult[] = [];
      for (const r of results) {
        const u = r.url?.trim();
        if (!u || seen.has(u)) continue;
        seen.add(u);
        deduped.push(r);
      }

      return deduped.slice(0, params.count).map((entry) => {
        const title = entry.title ?? "";
        const entryUrl = entry.url ?? "";
        const description = entry.content ?? "";
        return {
          title: title ? wrapWebContent(title, "web_search") : "",
          url: entryUrl,
          description: description ? wrapWebContent(description, "web_search") : "",
          published: entry.publishedDate || undefined,
          siteName: resolveSiteName(entryUrl) || undefined,
          engine: entry.engine || undefined,
        };
      });
    },
  );
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

function createSearxngToolDefinition(
  searchConfig: SearchConfigRecord | undefined,
  pluginConfig: SearxngPluginConfig,
): WebSearchProviderToolDefinition {
  return {
    description:
      "Search the web using SearXNG metasearch engine. Aggregates results from multiple search engines (Brave, DuckDuckGo, Startpage, etc.). Returns titles, URLs, and snippets.",
    parameters: createSearxngSchema(),
    execute: async (args) => {
      const baseUrl = resolveSearxngBaseUrl(pluginConfig);
      if (!baseUrl) {
        return {
          error: "missing_searxng_base_url",
          message:
            "web_search (searxng) needs a SearXNG instance URL. Set SEARXNG_BASE_URL in the Gateway environment, or configure plugins.entries.searxng.config.webSearch.baseUrl.",
          docs: "https://docs.openclaw.ai/tools/web",
        };
      }

      const apiKey = resolveSearxngApiKey(pluginConfig);
      const params = args as Record<string, unknown>;
      const query = readStringParam(params, "query", { required: true });
      const count =
        readNumberParam(params, "count", { integer: true }) ??
        searchConfig?.maxResults ??
        undefined;
      const language = readStringParam(params, "language");
      const categories = readStringParam(params, "categories");

      const cacheKey = buildSearchCacheKey([
        "searxng",
        query,
        resolveSearchCount(count, DEFAULT_SEARCH_COUNT),
        language,
        categories,
        baseUrl,
      ]);
      const cached = readCachedSearchPayload(cacheKey);
      if (cached) {
        return cached;
      }

      const start = Date.now();
      const results = await runSearxngSearch({
        query,
        baseUrl,
        apiKey,
        count: resolveSearchCount(count, DEFAULT_SEARCH_COUNT),
        timeoutSeconds: resolveSearchTimeoutSeconds(searchConfig),
        language: language ?? undefined,
        categories: categories ?? undefined,
      });

      const payload = {
        query,
        provider: "searxng",
        count: results.length,
        tookMs: Date.now() - start,
        externalContent: {
          untrusted: true,
          source: "web_search",
          provider: "searxng",
          wrapped: true,
        },
        results,
      };
      writeCachedSearchPayload(cacheKey, payload, resolveSearchCacheTtlMs(searchConfig));
      return payload;
    },
  };
}

// ---------------------------------------------------------------------------
// Provider export
// ---------------------------------------------------------------------------

export function createSearxngWebSearchProvider(): WebSearchProviderPlugin {
  return {
    id: "searxng",
    label: "SearXNG",
    hint: "Self-hosted metasearch · aggregates Brave, DuckDuckGo, Startpage, etc.",
    credentialLabel: "SearXNG Base URL",
    envVars: ["SEARXNG_BASE_URL"],
    placeholder: "http://localhost:8080",
    signupUrl: "https://docs.searxng.org/admin/installation.html",
    docsUrl: "https://docs.openclaw.ai/tools/web",
    autoDetectOrder: 60,
    credentialPath: "plugins.entries.searxng.config.webSearch.baseUrl",
    inactiveSecretPaths: ["plugins.entries.searxng.config.webSearch.apiKey"],
    getCredentialValue: (_searchConfig) => readProviderEnvValue(["SEARXNG_BASE_URL"]),
    setCredentialValue: (_searchConfig, _value) => {
      // SearXNG credential is the base URL, typically set via env var
    },
    getConfiguredCredentialValue: (config) =>
      resolveProviderWebSearchPluginConfig(config, "searxng")?.baseUrl as string | undefined,
    setConfiguredCredentialValue: (configTarget, value) => {
      setProviderWebSearchPluginConfigValue(configTarget, "searxng", "baseUrl", value);
    },
    createTool: (ctx) => {
      const pluginConfig = resolvePluginConfig(
        ctx.searchConfig as SearchConfigRecord | undefined,
        ctx.config as Record<string, unknown> | undefined,
      );
      return createSearxngToolDefinition(
        ctx.searchConfig as SearchConfigRecord | undefined,
        pluginConfig,
      );
    },
  };
}
