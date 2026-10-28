'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const readline = require('node:readline');
const { Bridge } = require('../bridge');

function fakeSpawn(handler) {
  return function spawnFn(command, args, options) {
    const proc = new EventEmitter();
    proc.stdin = new PassThrough();
    proc.stdout = new PassThrough();
    proc.stderr = new PassThrough();
    proc.kill = () => proc.emit('exit', 0);
    const rl = readline.createInterface({ input: proc.stdin });
    rl.on('line', (line) => {
      const request = JSON.parse(line);
      proc.stdout.write(JSON.stringify(handler(request)) + '\n');
    });
    return proc;
  };
}

test('bridge resolves a result by request id', async () => {
  const bridge = new Bridge('python', ['-m', 'edith.bridge'], {
    spawnFn: fakeSpawn((req) => ({ id: req.id, result: { ok: true, version: '0.1.0' } })),
  });
  const result = await bridge.request('ping', {});
  assert.deepEqual(result, { ok: true, version: '0.1.0' });
  bridge.close();
});

test('bridge rejects when the engine returns an error', async () => {
  const bridge = new Bridge('python', ['-m', 'edith.bridge'], {
    spawnFn: fakeSpawn((req) => ({ id: req.id, error: { message: 'boom' } })),
  });
  await assert.rejects(() => bridge.request('transcribe', {}), /boom/);
  bridge.close();
});

test('bridge rejects all pending requests if the process errors', async () => {
  function failingSpawn() {
    const proc = new EventEmitter();
    proc.stdin = new PassThrough();
    proc.stdout = new PassThrough();
    proc.stderr = new PassThrough();
    proc.kill = () => {};
    setImmediate(() => proc.emit('error', new Error('spawn failed')));
    return proc;
  }
  const bridge = new Bridge('python', ['-m', 'edith.bridge'], { spawnFn: failingSpawn });
  await assert.rejects(() => bridge.request('ping', {}), /spawn failed/);
  bridge.close();
});
