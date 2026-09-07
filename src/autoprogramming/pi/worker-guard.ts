/**
 * Cooperative root guard for implementation-only Pi workers.
 *
 * This blocks ordinary accidental traversal through built-in tools. It is not
 * advertised as an OS security boundary: strict runs additionally need a
 * sandboxed BashOperations adapter (bubblewrap/sandbox-runtime).
 */
import fs from "node:fs";
import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function inside(root: string, requested: string): boolean {
  const resolved = path.resolve(root, requested.replace(/^@/, ""));
  let checked = resolved;
  try {
    checked = fs.realpathSync(resolved);
  } catch {
    try {
      checked = path.join(fs.realpathSync(path.dirname(resolved)), path.basename(resolved));
    } catch {
      // path.resolve result is still checked below
    }
  }
  return checked === root || checked.startsWith(root + path.sep);
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", (event, ctx) => {
    const root = fs.realpathSync(ctx.cwd);
    if (["read", "write", "edit", "grep", "find", "ls"].includes(event.toolName)) {
      const requested = String(event.input.path ?? ".");
      if (!inside(root, requested)) {
        return { block: true, reason: "Implementation workers may access only their task directory." };
      }
    }
    if (event.toolName === "bash") {
      const command = String(event.input.command ?? "");
      // Prevent common traversal/discovery paths. This is defense in depth; the
      // strict controller never treats shell-string filtering as a sandbox.
      if (/(^|[\s;&|])\.\.\//.test(command) || /\$HOME|~\/|\bprintenv\b|\benv\s*$/.test(command)) {
        return { block: true, reason: "Shell traversal and environment inspection are unavailable in this task." };
      }
      const absoluteToken = command.match(/(^|[\s;&|>])\/(?!\/)([^\s;&|]*)/);
      if (absoluteToken && !absoluteToken[0].includes("://")) {
        return { block: true, reason: "Use paths relative to the implementation task directory." };
      }
    }
  });
}
