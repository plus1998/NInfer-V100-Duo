import { createServer } from 'node:http';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { Codex } from '@openai/codex-sdk';
import { query } from '@anthropic-ai/claude-agent-sdk';
import { AuthStorage, createAgentSession, SessionManager } from '@mariozechner/pi-coding-agent';
import { getModel } from '@mariozechner/pi-ai';
import { WebSocketServer } from 'ws';

const outputDir = resolve(process.argv[2] ?? '/tmp/ninfer-agent-prefill');
const prompt = process.argv[3] ??
  'Inspect a C++ repository and explain how its bounded task queue wakes waiting producers on close.';
const repoRoot = resolve(import.meta.dirname, '../../..');
if (outputDir === repoRoot || outputDir.startsWith(repoRoot + '/')) {
  throw new Error('Choose an output directory outside the repository; captured prompts are private');
}
await mkdir(outputDir, { recursive: true });
const workspace = await mkdtemp(resolve(tmpdir(), 'ninfer-agent-empty-'));
const requests = [];
const server = createServer(async (request, response) => {
  let body = '';
  for await (const chunk of request) body += chunk.toString();
  if (body) {
    try { requests.push({ path: request.url, payload: JSON.parse(body) }); }
    catch {}
  }
  response.writeHead(400, { 'content-type': 'application/json' });
  response.end(JSON.stringify({ type: 'error', error: {
    type: 'invalid_request_error', message: 'Local capture only: no model inference',
  } }));
});
const sockets = new WebSocketServer({ noServer: true });
server.on('upgrade', (request, socket, head) => {
  sockets.handleUpgrade(request, socket, head, connection => {
    connection.on('message', data => {
      try { requests.push({ path: request.url, payload: JSON.parse(data.toString()) }); }
      catch { return; }
      connection.send(JSON.stringify({ type: 'error', error: {
        type: 'invalid_request_error', message: 'Local capture only: no model inference',
      } }));
      connection.close();
    });
  });
});
await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
const baseUrl = `http://127.0.0.1:${server.address().port}`;

const capture = async (name, run) => {
  const start = requests.length;
  try { await run(); } catch (error) { console.error(`${name}: ${String(error).slice(0, 160)}`); }
  const candidates = requests.slice(start).filter(item =>
    item.path.startsWith('/v1/messages') || item.path.startsWith('/v1/responses'));
  const selected = candidates.sort((left, right) =>
    JSON.stringify(right.payload).length - JSON.stringify(left.payload).length)[0];
  if (!selected) throw new Error(`${name}: SDK produced no model request`);
  await writeFile(resolve(outputDir, `${name}.request.json`),
    JSON.stringify(selected.payload, null, 2), { mode: 0o600 });
  const parts = [];
  const payload = selected.payload;
  if (payload.system) parts.push(JSON.stringify(payload.system));
  if (payload.instructions) parts.push(JSON.stringify(payload.instructions));
  if (payload.tools) parts.push(JSON.stringify(payload.tools));
  if (payload.messages) parts.push(JSON.stringify(payload.messages));
  if (payload.input) parts.push(JSON.stringify(payload.input));
  const text = parts.join('\n\n');
  await writeFile(resolve(outputDir, `${name}.prompt.txt`), text, { mode: 0o600 });
  console.log(`${name}: ${selected.path}, ${Buffer.byteLength(text)} bytes of captured prompt/tools`);
};

try {
  await capture('pi', async () => {
    const authStorage = AuthStorage.inMemory();
    authStorage.setRuntimeApiKey('anthropic', 'local-capture-only');
    const model = { ...getModel('anthropic', 'claude-sonnet-4-5'), baseUrl };
    const { session } = await createAgentSession({ cwd: workspace, agentDir: workspace,
      sessionManager: SessionManager.inMemory(workspace), authStorage, model, thinkingLevel: 'off' });
    try { await session.prompt(prompt); } finally { await session.dispose(); }
  });
  await capture('codex', async () => {
    const codex = new Codex({ baseUrl: `${baseUrl}/v1`, apiKey: 'local-capture-only',
      env: { PATH: process.env.PATH, HOME: workspace, OPENAI_API_KEY: 'local-capture-only' } });
    const thread = codex.startThread({ workingDirectory: workspace, skipGitRepoCheck: true,
      sandboxMode: 'read-only', approvalPolicy: 'never', networkAccessEnabled: false,
      webSearchMode: 'disabled' });
    await thread.run(prompt);
  });
  await capture('claude', async () => {
    const stream = query({ prompt, options: { cwd: workspace, persistSession: false,
      settingSources: [], maxTurns: 1, tools: { type: 'preset', preset: 'claude_code' },
      systemPrompt: { type: 'preset', preset: 'claude_code' },
      env: { PATH: process.env.PATH, HOME: workspace,
        ANTHROPIC_BASE_URL: baseUrl, ANTHROPIC_API_KEY: 'local-capture-only' },
    } });
    try { for await (const event of stream) { if (event.type === 'result') break; } }
    finally { stream.close(); }
  });
} finally {
  sockets.close();
  server.close();
  await rm(workspace, { recursive: true, force: true });
}
