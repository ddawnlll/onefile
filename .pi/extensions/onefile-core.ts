/** ONEFILE project checks. Advisory policy, NOT a sandbox or OOS access-control boundary. */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export function isOnefile(root: string): boolean {
  return existsSync(resolve(root, "one.py")) && existsSync(resolve(root, "pyproject.toml"));
}

export function sourceSha256(root: string): string | null {
  const file = resolve(root, "one.py");
  return existsSync(file) ? createHash("sha256").update(readFileSync(file)).digest("hex") : null;
}

export function evidenceStatus(root: string): { codeSha: string | null; evidenceSha: string | null; match: boolean | null } {
  const codeSha = sourceSha256(root);
  const path = resolve(root, "brick1_evidence.json");
  if (!existsSync(path)) return { codeSha, evidenceSha: null, match: null };
  try {
    const record = JSON.parse(readFileSync(path, "utf8"));
    const evidenceSha = record.code_sha256 ?? record.b1_data_readiness?.code_sha256 ?? null;
    return { codeSha, evidenceSha, match: typeof evidenceSha === "string" && codeSha !== null ? evidenceSha === codeSha : null };
  } catch {
    return { codeSha, evidenceSha: null, match: null };
  }
}

/** Explicit human opt-in requires ONEFILE_ALLOW_OOS=1 before the Pi process starts. */
export function forbiddenOosCommand(command: string, allow = process.env.ONEFILE_ALLOW_OOS === "1"): boolean {
  return !allow && /--unlock-oos\b/i.test(command);
}

export default function onefileCore(pi: ExtensionAPI): void {
  // The same failing command is not useful three times on an unchanged source revision.
  const failures = new Map<string, number>();

  pi.on("session_start", (_event, ctx) => {
    if (!isOnefile(ctx.cwd)) return;
    const status = evidenceStatus(ctx.cwd);
    if (status.match === false && ctx.hasUI) {
      ctx.ui.notify("ONEFILE: brick1_evidence.json source hash differs from one.py; results are historical, not current proof.", "warning");
    }
  });

  pi.on("tool_call", (event, ctx) => {
    if (!isOnefile(ctx.cwd)) return;
    if (event.toolName !== "bash" && event.toolName !== "powershell" && event.toolName !== "codemode") return;
    const input = event.input as Record<string, unknown>;
    const command = typeof input.command === "string" ? input.command : typeof input.code === "string" ? input.code : typeof input.script === "string" ? input.script : "";
    if (forbiddenOosCommand(command)) {
      return { block: true, reason: "ONEFILE sealed OOS access requires explicit human authorization and an isolated OOS environment. Do not pass --unlock-oos in the DEV session." };
    }
    if (event.toolName === "bash" || event.toolName === "powershell") {
      const key = `${sourceSha256(ctx.cwd)}:${command}`;
      if ((failures.get(key) ?? 0) >= 2) {
        return { block: true, reason: "ONEFILE loop breaker: this identical command already failed twice against the same source; diagnose or change the approach first." };
      }
    }
  });

  pi.on("tool_result", (event, ctx) => {
    if (!isOnefile(ctx.cwd) || (event.toolName !== "bash" && event.toolName !== "powershell")) return;
    const command = typeof event.input.command === "string" ? event.input.command : "";
    if (!command) return;
    const key = `${sourceSha256(ctx.cwd)}:${command}`;
    const details = event.details as { exitCode?: number; exit_code?: number } | undefined;
    const nonzero = details?.exitCode !== undefined ? details.exitCode !== 0 : details?.exit_code !== undefined ? details.exit_code !== 0 : false;
    if (event.isError || nonzero) failures.set(key, (failures.get(key) ?? 0) + 1);
    else failures.delete(key);
  });

  pi.registerCommand("onefile-status", {
    description: "Show ONEFILE source SHA and whether recorded evidence matches it",
    handler: async (_args, ctx) => {
      if (!isOnefile(ctx.cwd)) { ctx.ui.notify("Not in a ONEFILE checkout", "warning"); return; }
      const status = evidenceStatus(ctx.cwd);
      ctx.ui.notify(`Source SHA-256: ${status.codeSha ?? "missing"}\nEvidence SHA-256: ${status.evidenceSha ?? "missing"}\nCurrent evidence match: ${status.match === null ? "UNKNOWN" : status.match ? "YES" : "NO (historical)"}`, status.match === false ? "warning" : "info");
    },
  });
}