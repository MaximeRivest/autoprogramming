/** Current-source web research tools for the strategy-only Pi process. */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

function publicHttpUrl(raw: string): URL {
  const url = new URL(raw);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Only public HTTP(S) research URLs are allowed.');
  }
  const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  const privateHost =
    host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') ||
    host === '0.0.0.0' || host === '::' || host === '::1' ||
    /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) ||
    /^169\.254\./.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host) ||
    /^fc/i.test(host) || /^fd/i.test(host) || /^fe[89ab]/i.test(host);
  if (privateHost) throw new Error('Private/local research URLs are not allowed.');
  return url;
}

function compact(value: string): string {
  return value.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ").trim();
}

function unwrap(raw: string): string {
  let value = raw.replaceAll("&amp;", "&");
  if (value.startsWith("//")) value = "https:" + value;
  try {
    const parsed = new URL(value);
    if (parsed.hostname.endsWith("duckduckgo.com")) {
      return decodeURIComponent(parsed.searchParams.get("uddg") || "");
    }
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.toString() : "";
  } catch {
    return "";
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "web_search",
    label: "Web Search",
    description: "Search current public web sources before planning implementation approaches. Returns titles, URLs, and snippets.",
    parameters: Type.Object({
      query: Type.String({ description: "Task-specific technology/model/algorithm query; never include private examples" }),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
    }),
    async execute(_id, params, signal) {
      const limit = params.limit ?? 6;
      const response = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(params.query)}`, {
        headers: { "user-agent": "Mozilla/5.0 (compatible; AutoProgrammingResearch/0.2)" },
        signal,
      });
      if (!response.ok) throw new Error(`web search HTTP ${response.status}`);
      const body = await response.text();
      const anchors = [...body.matchAll(/<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi)];
      const snippets = [...body.matchAll(/<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div)>/gi)];
      const results: Array<{ title: string; url: string; snippet: string }> = [];
      const seen = new Set<string>();
      for (let i = 0; i < anchors.length && results.length < limit; i++) {
        const rawUrl = unwrap(anchors[i][1]);
        let url: string;
        try { url = publicHttpUrl(rawUrl).toString(); } catch { continue; }
        if (seen.has(url)) continue;
        seen.add(url);
        results.push({ title: compact(anchors[i][2]), url, snippet: compact(snippets[i]?.[1] || "") });
      }
      if (results.length === 0) throw new Error("search returned no parseable current sources");
      return {
        content: [{ type: "text", text: JSON.stringify({ query: params.query, searchedAt: new Date().toISOString(), results }, null, 2) }],
        details: { query: params.query, results },
      };
    },
  });

  pi.registerTool({
    name: "web_fetch",
    label: "Web Fetch",
    description: "Fetch a public source selected from web_search for closer inspection.",
    parameters: Type.Object({ url: Type.String() }),
    async execute(_id, params, signal) {
      const parsed = publicHttpUrl(params.url);
      const response = await fetch(parsed, { signal, headers: { "user-agent": "AutoProgrammingResearch/0.2" } });
      if (!response.ok) throw new Error(`fetch HTTP ${response.status}`);
      const finalUrl = publicHttpUrl(response.url);
      const text = compact((await response.text()).slice(0, 250_000)).slice(0, 45_000);
      return { content: [{ type: "text", text: `${finalUrl.toString()}\n\n${text}` }], details: { url: finalUrl.toString() } };
    },
  });
}
