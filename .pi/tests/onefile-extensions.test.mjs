import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { sourceSha256, evidenceStatus, forbiddenOosCommand, isOnefile, default as onefileCore } from '../extensions/onefile-core.ts';
import { revision, verify, default as onefileVerify } from '../extensions/onefile-verify.ts';

function fixture(fn) {
  const root = mkdtempSync(join(tmpdir(), 'onefile-extension-test-'));
  try {
    writeFileSync(join(root, 'one.py'), 'print("hello")\n');
    writeFileSync(join(root, 'pyproject.toml'), '[project]\nname="onefile"\n');
    writeFileSync(join(root, 'uv.lock'), '# lock\n');
    return fn(root);
  } finally { rmSync(root, { recursive: true, force: true }); }
}
function stub() {
  const hooks = new Map(); const commands = new Map();
  return { hooks, commands,
    api: { on(name, handler) { hooks.set(name, handler); }, registerCommand(name, cfg) { commands.set(name, cfg); } } };
}

test('source SHA and evidence mismatch are detected without mutating evidence', () => fixture(root => {
  assert.equal(isOnefile(root), true);
  assert.match(sourceSha256(root), /^[a-f0-9]{64}$/);
  assert.equal(evidenceStatus(root).match, null);
  writeFileSync(join(root, 'brick1_evidence.json'), JSON.stringify({ code_sha256: '0'.repeat(64) }));
  assert.equal(evidenceStatus(root).match, false);
  writeFileSync(join(root, 'brick1_evidence.json'), JSON.stringify({ code_sha256: sourceSha256(root) }));
  assert.equal(evidenceStatus(root).match, true);
}));

test('OOS flag blocked without environment opt-in, including nested bash calls', () => fixture(root => {
  assert.equal(forbiddenOosCommand('uv run python one.py --unlock-oos', false), true);
  assert.equal(forbiddenOosCommand('uv run python one.py --unlock-oos', true), false);
  const { hooks, api } = stub(); onefileCore(api);
  const ctx = { cwd: root };
  assert.equal(hooks.get('tool_call')({ toolName: 'bash', input: { command: 'uv run python one.py --unlock-oos' }, parentToolCallId: 'code/1' }, ctx)?.block, true);
  assert.equal(hooks.get('tool_call')({ toolName: 'bash', input: { command: 'uv run python one.py --self-test' } }, ctx), undefined);
}));

test('loop breaker avoids a third exact failure until source changes', () => fixture(root => {
  const { hooks, api } = stub(); onefileCore(api);
  const ctx = { cwd: root };
  const c = 'uv run python one.py --self-test';
  const call = () => hooks.get('tool_call')({ toolName:'bash', input:{ command:c } },ctx);
  const failed = () => hooks.get('tool_result')({ toolName:'bash', input:{command:c}, isError:true },ctx);
  assert.equal(call(), undefined); failed(); assert.equal(call(), undefined); failed();
  assert.equal(call()?.block, true);
  writeFileSync(join(root,'one.py'),'print("new")\n');
  assert.equal(call(), undefined);
}));

test('revision reflects source, project metadata and lockfile changes', () => fixture(root => {
  const before = revision(root); assert.match(before, /^[a-f0-9]{64}$/);
  writeFileSync(join(root, 'uv.lock'), '# updated lock\n');
  assert.notEqual(before, revision(root));
}));

test('verify reports missing environment and does not claim PASS', async () => {
  const root = mkdtempSync(join(tmpdir(), 'onefile-verify-missing-'));
  try {
    const result = await verify(root);
    assert.equal(result.passed, false);
    assert.equal(result.current, false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('verification is manual only: no edit, settlement, or other automatic hooks', () => {
  const { hooks, commands, api } = stub(); onefileVerify(api);
  assert.equal(hooks.size, 0);
  assert.equal(typeof commands.get('onefile-verify')?.handler, 'function');
});

test('manual verification runs the self-test once and rejects failure or stale code', async () => {
  const root = mkdtempSync(join(tmpdir(), 'onefile-verify-result-'));
  try {
    writeFileSync(join(root, 'one.py'), '# before\n');
    writeFileSync(join(root, 'pyproject.toml'), '# project\n');
    writeFileSync(join(root, 'uv.lock'), '# lock\n');
    const calls = [];
    const run = async (cwd, args) => {
      calls.push({ cwd, args });
      return { command: 'uv ' + args.join(' '), exitCode: 0, output: 'passed', durationMs: 1 };
    };
    assert.equal((await verify(root, run)).passed, true);
    assert.deepEqual(calls, [{ cwd: root, args: ['run', '--no-sync', 'python', 'one.py', '--self-test'] }]);
    assert.equal((await verify(root, async () => ({ exitCode: 1 }))).passed, false);
    const stale = await verify(root, async () => {
      writeFileSync(join(root, 'one.py'), '# changed during check\n');
      return { exitCode: 0 };
    });
    assert.equal(stale.current, false);
    assert.equal(stale.passed, false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
