/** Explicit ONEFILE self-test. Never schedules work after edits or at agent settlement. */
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export function revision(root: string): string | null {
  const hash = createHash("sha256");
  for (const filename of ["one.py", "pyproject.toml", "uv.lock"]) {
    const path = resolve(root, filename);
    if (!existsSync(path)) return null;
    hash.update(filename).update("\0").update(readFileSync(path)).update("\0");
  }
  return hash.digest("hex");
}

export type Check = { command: string; exitCode: number | null; output: string; durationMs: number };
export type Verification = { revision: string | null; current: boolean; passed: boolean; checks: Check[] };

function execute(root: string, args: string[], timeoutMs = 120_000): Promise<Check> {
  return new Promise((done) => {
    const started = Date.now();
    let output = "";
    let completed = false;
    let child: ReturnType<typeof spawn>;
    const finish = (exitCode: number | null) => {
      if (completed) return;
      completed = true;
      clearTimeout(timeout);
      done({ command: `uv ${args.join(" ")}`, exitCode, output: output.slice(-5000), durationMs: Date.now() - started });
    };
    try {
      child = spawn("uv", args, { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, UV_NO_PROGRESS: "1" } });
    } catch (error) {
      done({ command: `uv ${args.join(" ")}`, exitCode: null, output: String(error), durationMs: Date.now() - started });
      return;
    }
    const collect = (b: Buffer) => { output = (output + b.toString()).slice(-8000); };
    child.stdout?.on("data", collect);
    child.stderr?.on("data", collect);
    child.on("error", (err) => { output += `\n${err.message}`; finish(null); });
    child.on("close", (code) => finish(code));
    const timeout = setTimeout(() => { output += "\nTIMEOUT"; child.kill("SIGTERM"); finish(null); }, timeoutMs);
    timeout.unref();
  });
}

/** Runs only the existing synthetic checks; does not prove strategy profitability or OOS validity. */
export async function verify(root: string, run = execute): Promise<Verification> {
  const before = revision(root);
  if (!before) return { revision: before, current: false, passed: false, checks: [] };
  // Running the script already parses one.py; a separate py_compile process is redundant.
  const checks = [await run(root, ["run", "--no-sync", "python", "one.py", "--self-test"])];
  const current = revision(root) === before;
  return { revision: before, current, passed: current && checks.every((check) => check.exitCode === 0), checks };
}

export default function onefileVerify(pi: ExtensionAPI): void {
  pi.registerCommand("onefile-verify", {
    description: "Run ONEFILE synthetic regression checks on demand for the current code/dependency revision",
    handler: async (_args, ctx) => {
      if (!revision(ctx.cwd)) { ctx.ui.notify("ONEFILE files not found", "warning"); return; }
      const result = await verify(ctx.cwd);
      ctx.ui.notify(`ONEFILE ${result.passed ? "PASS" : "FAIL / STALE"} (${result.revision?.slice(0, 12)})\n${result.checks.map(c => `${c.exitCode === 0 ? "PASS" : "FAIL"}: ${c.command} (${c.durationMs}ms)`).join("\n")}`, result.passed ? "info" : "warning");
    },
  });

}
