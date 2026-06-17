/**
 * OpenAI-compatible native PDF via Chat Completions API.
 * Supports both standard OpenAI and Azure OpenAI (auth header differs).
 */

import { isRecord } from "../../utils.js";
import { normalizeSecretInput } from "../../utils/normalize-secret-input.js";

type PdfInput = {
  base64: string;
  filename?: string;
};

type OpenAIPdfContentPart =
  | { type: "file"; file: { filename: string; file_data: string } }
  | { type: "text"; text: string };

type OpenAIChoice = {
  message?: { content?: string };
};

export async function openaiAnalyzePdf(params: {
  apiKey: string;
  modelId: string;
  prompt: string;
  pdfs: PdfInput[];
  maxTokens?: number;
  baseUrl?: string;
  /** Model API identifier – determines auth header style. */
  api?: string;
}): Promise<string> {
  const apiKey = normalizeSecretInput(params.apiKey);
  if (!apiKey) {
    throw new Error("OpenAI PDF: apiKey required");
  }

  const content: OpenAIPdfContentPart[] = [];
  for (const pdf of params.pdfs) {
    content.push({
      type: "file",
      file: {
        filename: pdf.filename ?? "document.pdf",
        file_data: `data:application/pdf;base64,${pdf.base64}`,
      },
    });
  }
  content.push({ type: "text", text: params.prompt });

  const isAzure = params.api?.startsWith("azure");
  const baseUrl = (params.baseUrl ?? "https://api.openai.com/v1").replace(/\/+$/, "");
  const url = `${baseUrl}/chat/completions`;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (isAzure) {
    headers["api-key"] = apiKey;
  } else {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: params.modelId,
      max_completion_tokens: params.maxTokens ?? 4096,
      messages: [{ role: "user", content }],
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `OpenAI PDF request failed (${res.status} ${res.statusText})${body ? `: ${body.slice(0, 400)}` : ""}`,
    );
  }

  const json = (await res.json().catch(() => null)) as unknown;
  if (!isRecord(json)) {
    throw new Error("OpenAI PDF response was not JSON.");
  }

  const choices = json.choices as OpenAIChoice[] | undefined;
  if (!Array.isArray(choices) || choices.length === 0) {
    throw new Error("OpenAI PDF returned no choices.");
  }

  const text = choices[0].message?.content ?? "";
  if (!text.trim()) {
    throw new Error("OpenAI PDF returned no text.");
  }

  return text.trim();
}
