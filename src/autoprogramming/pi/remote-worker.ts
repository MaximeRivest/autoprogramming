/**
 * Explicit remote tool transport for implementation workers.
 *
 * Loaded only when the user supplied RemoteCompute. Pi/model inference stays
 * local (and keeps using Pi OAuth); read/write/edit/bash operate in the staged
 * remote task directory so model setup and experiments do not load the local
 * machine. The Python controller synchronizes the isolated bundle before and
 * after the turn.
 */
import { spawn } from "node:child_process";
import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  type BashOperations,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  type EditOperations,
  type ReadOperations,
  type WriteOperations,
} from "@earendil-works/pi-coding-agent";

function q(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

function required(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Remote worker extension requires ${name}`);
  return value;
}

function sshExec(remote: string, command: string, input?: Buffer): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn("ssh", [remote, command], { stdio: ["pipe", "pipe", "pipe"] });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    child.stdout.on("data", (data) => out.push(data));
    child.stderr.on("data", (data) => err.push(data));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) reject(new Error(`SSH failed (${code}): ${Buffer.concat(err).toString()}`));
      else resolve(Buffer.concat(out));
    });
    if (input) child.stdin.end(input);
    else child.stdin.end();
  });
}

function mapPath(localCwd: string, remoteCwd: string, requested: string): string {
  const raw = requested.replace(/^@/, "");
  const local = path.resolve(localCwd, raw);
  const relative = path.relative(localCwd, local);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error("Remote worker paths must remain inside the task root");
  }
  return relative ? path.posix.join(remoteCwd, relative.split(path.sep).join("/")) : remoteCwd;
}

function readOps(remote: string, remoteCwd: string, localCwd: string): ReadOperations {
  return {
    readFile: (p) => sshExec(remote, `cat ${q(mapPath(localCwd, remoteCwd, p))}`),
    access: (p) => sshExec(remote, `test -r ${q(mapPath(localCwd, remoteCwd, p))}`).then(() => {}),
    detectImageMimeType: async (p) => {
      try {
        const result = await sshExec(remote, `file --mime-type -b ${q(mapPath(localCwd, remoteCwd, p))}`);
        const mime = result.toString().trim();
        return ["image/jpeg", "image/png", "image/gif", "image/webp"].includes(mime) ? mime : null;
      } catch {
        return null;
      }
    },
  };
}

function writeOps(remote: string, remoteCwd: string, localCwd: string): WriteOperations {
  return {
    writeFile: async (p, content) => {
      const target = mapPath(localCwd, remoteCwd, p);
      const encoded = Buffer.from(content).toString("base64");
      await sshExec(
        remote,
        `mkdir -p ${q(path.posix.dirname(target))} && base64 -d > ${q(target)}`,
        Buffer.from(encoded),
      );
    },
    mkdir: (p) => sshExec(remote, `mkdir -p ${q(mapPath(localCwd, remoteCwd, p))}`).then(() => {}),
  };
}

function editOps(remote: string, remoteCwd: string, localCwd: string): EditOperations {
  const r = readOps(remote, remoteCwd, localCwd);
  const w = writeOps(remote, remoteCwd, localCwd);
  return { readFile: r.readFile, access: r.access, writeFile: w.writeFile };
}

function bashOps(remote: string, remoteCwd: string, environmentPrefix: string): BashOperations {
  return {
    exec: (command, _cwd, { onData, signal, timeout }) =>
      new Promise((resolve, reject) => {
        const exported = environmentPrefix ? `export ${environmentPrefix}; ` : "";
        const wrapped = `cd ${q(remoteCwd)} && ${exported}${command}`;
        const child = spawn("ssh", [remote, wrapped], { stdio: ["ignore", "pipe", "pipe"] });
        let timedOut = false;
        const timer = timeout
          ? setTimeout(() => {
              timedOut = true;
              child.kill("SIGTERM");
            }, timeout * 1000)
          : undefined;
        child.stdout.on("data", onData);
        child.stderr.on("data", onData);
        child.on("error", reject);
        const abort = () => child.kill("SIGTERM");
        signal?.addEventListener("abort", abort, { once: true });
        child.on("close", (code) => {
          if (timer) clearTimeout(timer);
          signal?.removeEventListener("abort", abort);
          if (signal?.aborted) reject(new Error("aborted"));
          else if (timedOut) reject(new Error(`timeout:${timeout}`));
          else resolve({ exitCode: code });
        });
      }),
  };
}

export default function (pi: ExtensionAPI) {
  const remote = required("AP_REMOTE_ENDPOINT");
  const remoteCwd = required("AP_REMOTE_CWD");
  const environmentPrefix = process.env.AP_REMOTE_ENV_PREFIX?.trim() || "";
  const localCwd = process.cwd();

  pi.registerTool(createReadTool(localCwd, { operations: readOps(remote, remoteCwd, localCwd) }));
  pi.registerTool(createWriteTool(localCwd, { operations: writeOps(remote, remoteCwd, localCwd) }));
  pi.registerTool(createEditTool(localCwd, { operations: editOps(remote, remoteCwd, localCwd) }));
  pi.registerTool(createBashTool(localCwd, {
    operations: bashOps(remote, remoteCwd, environmentPrefix),
    exposeSessionEnvironment: false,
  }));

  pi.on("before_agent_start", (event) => ({
    systemPrompt: event.systemPrompt.replace(
      `Current working directory: ${localCwd}`,
      `Current working directory: ${remoteCwd} (user-provided remote compute)`,
    ),
  }));
}
