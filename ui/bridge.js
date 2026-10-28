'use strict';

// A tiny line-delimited JSON bridge to the Python engine.
//
// The UI spawns `python -m edith.bridge` once and sends one JSON request per
// line; each response line is matched back to its request by id. This module
// has no Electron dependency so it can be unit-tested with plain `node --test`.

const { spawn } = require('node:child_process');
const readline = require('node:readline');

class Bridge {
  /**
   * @param {string} command  e.g. path to .venv/Scripts/python.exe
   * @param {string[]} args   e.g. ['-m', 'edith.bridge']
   * @param {object} [options] spawn options plus an optional `spawnFn` for tests
   */
  constructor(command, args, options = {}) {
    const { spawnFn = spawn, ...spawnOptions } = options;
    this.command = command;
    this.args = args;
    this.spawnOptions = spawnOptions;
    this.spawnFn = spawnFn;
    this.proc = null;
    this.rl = null;
    this.nextId = 1;
    this.pending = new Map();
    this.stderr = '';
    this._start();
  }

  _start() {
    this.proc = this.spawnFn(this.command, this.args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      ...this.spawnOptions,
    });

    this.rl = readline.createInterface({ input: this.proc.stdout });
    this.rl.on('line', (line) => this._onLine(line));
    this.proc.stderr.on('data', (chunk) => {
      this.stderr += chunk.toString();
    });
    this.proc.on('error', (err) => this._failAll(err));
    this.proc.on('exit', (code) => {
      if (code !== 0) this._failAll(new Error(`bridge exited with code ${code}`));
    });
  }

  _onLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      return; // ignore non-JSON noise on stdout
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.error) {
      pending.reject(new Error(message.error.message || 'bridge error'));
    } else {
      pending.resolve(message.result);
    }
  }

  _failAll(err) {
    for (const pending of this.pending.values()) pending.reject(err);
    this.pending.clear();
  }

  /**
   * Send one request and resolve with the result (or reject with an Error).
   * @param {string} method
   * @param {object} params
   * @returns {Promise<any>}
   */
  request(method, params = {}) {
    if (!this.proc || !this.proc.stdin || !this.proc.stdin.writable) {
      return Promise.reject(new Error('bridge process is not running'));
    }
    const id = this.nextId++;
    const payload = { id, method, params };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(JSON.stringify(payload) + '\n', (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  close() {
    try {
      if (this.proc && this.proc.stdin) this.proc.stdin.end();
    } catch {
      /* ignore */
    }
    try {
      if (this.proc) this.proc.kill();
    } catch {
      /* ignore */
    }
    if (this.rl) this.rl.close();
  }
}

module.exports = { Bridge };
