/** ONEFILE revision-bound checks. Background smoke check is advisory; final check is authoritative. */
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
export async function verify(root: string): Promise<Verification> {
  const before = revision(root);
  if (!before) return { revision: before, current: false, passed: false, checks: [] };
  const checks: Check[] = [];
  for (const args of [
    ["run", "--no-sync", "python", "-m", "py_compile", "one.py"],
    ["run", "--no-sync", "python", "one.py", "--self-test"],
  ]) {
    checks.push(await execute(root, args));
    if (checks[checks.length - 1].exitCode !== 0) break;
  }
  const current = revision(root) === before;
  return { revision: before, current, passed: current && checks.length === 2 && checks.every((check) => check.exitCode === 0), checks };
}

export default function onefileVerify(pi: ExtensionAPI): void {
  let startRevision: string | null = null;
  let active: Promise<Verification> | null = null;
  let activeRevision: string | null = null;
  let debounce: ReturnType<typeof setTimeout> | null = null;
  let lastFailedRevision: string | null = null;

  const checked = async (root: string) => {
    const currentRevision = revision(root);
    if (active && activeRevision === currentRevision) return active;
    activeRevision = currentRevision;
    const job = verify(root);
    active = job;
    return job;
  };

  pi.on("agent_start", (_event, ctx) => {
    if (existsSync(resolve(ctx.cwd, "one.py"))) startRevision = revision(ctx.cwd);
  });

  // Debounced best-effort background validation after direct file tool edits.
  // Never trust this until the final revision is rechecked at settlement.
  pi.on("tool_result", (event, ctx) => {
    if (event.isError || (event.toolName !== "edit" && event.toolName !== "write")) return;
    const path = typeof event.input.path === "string" ? resolve(ctx.cwd, event.input.path) : "";
    if (!["one.py", "pyproject.toml", "uv.lock"].some((name) => path === resolve(ctx.cwd, name))) return;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => { debounce = null; void checked(ctx.cwd).catch(() => {}); }, 650);
  });

  pi.on("agent_before_settle", async (event, ctx) => {
    if (event.outcome !== "completed" || !startRevision) return;
    if (debounce) { clearTimeout(debounce); debounce = null; }
    const latest = revision(ctx.cwd);
    if (!latest || latest === startRevision) return;
    const result = await checked(ctx.cwd);
    if (result.passed && result.current && result.revision === revision(ctx.cwd)) return;
    if (lastFailedRevision === latest || !event.context.canContinue) {
      if (ctx.hasUI) ctx.ui.notify("ONEFILE verification failed or stale; do not claim verified completion. Run /onefile-verify.", "warning");
      return;
    }
    lastFailedRevision = latest;
    const report = result.checks.map(c => `${c.command}: ${c.exitCode === 0 ? "PASS" : "FAIL"}\n${c.output}`).join("\n");
    return {
      entries: [{ type: "custom_message" as const, customType: "onefile-verify", display: false, content: `ONEFILE final-revision verification failed or became stale. Fix or explain; do not claim success. Revision: ${latest}\n${report || "No checks ran; missing locked environment files."}` }],
      continue: true,
    };
  });

  pi.registerCommand("onefile-verify", {
    description: "Run ONEFILE compile and synthetic regression checks for the current code/dependency revision",
    handler: async (_args, ctx) => {
      if (!revision(ctx.cwd)) { ctx.ui.notify("ONEFILE files not found", "warning"); return; }
      const result = await checked(ctx.cwd);
      ctx.ui.notify(`ONEFILE ${result.passed ? "PASS" : "FAIL / STALE"} (${result.revision?.slice(0, 12)})\n${result.checks.map(c => `${c.exitCode === 0 ? "PASS" : "FAIL"}: ${c.command} (${c.durationMs}ms)`).join("\n")}`, result.passed ? "info" : "warning");
    },
  });

  pi.on("session_shutdown", () => { if (debounce) clearTimeout(debounce); });
}