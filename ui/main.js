'use strict';

// Edith UI main process. It owns the Python bridge process and exposes a small,
// allow-listed IPC surface to the renderer. Security settings:
//   - contextIsolation: true
//   - nodeIntegration: false
//   - sandbox: true
//   - a preload script is the only bridge to the renderer.

const { app, BrowserWindow, dialog, ipcMain, session } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { Bridge } = require('./bridge');

let bridge = null;
let mainWindow = null;

function projectRoot() {
  return path.resolve(__dirname, '..');
}

function pythonExe() {
  const venv = path.join(projectRoot(), '.venv', 'Scripts', 'python.exe');
  return fs.existsSync(venv) ? venv : 'python';
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 820,
    title: 'Edith',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  // Only the microphone permission is allowed; everything else is denied.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media');
  });

  bridge = new Bridge(pythonExe(), ['-m', 'edith.bridge'], {
    cwd: projectRoot(),
  });
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (bridge) bridge.close();
});

// ---- IPC allow-list -------------------------------------------------------

ipcMain.handle('edith:ping', () => bridge.request('ping', {}));

ipcMain.handle('edith:transcribe', (_event, payload) => {
  const { audio, model, device, language } = payload || {};
  return bridge.request('transcribe', {
    audio,
    model: model || 'large-v3-turbo',
    device: device || 'auto',
    language: language || 'en',
  });
});

ipcMain.handle('edith:pickAudioFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose an audio file',
    properties: ['openFile'],
    filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg'] }],
  });
  return canceled ? null : filePaths[0];
});

// The renderer records a WAV in memory; the main process saves it next to the
// project so the Python engine can load it by path.
ipcMain.handle('edith:saveRecording', (_event, bytes) => {
  const dir = path.join(projectRoot(), 'output');
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, 'ui-recording.wav');
  fs.writeFileSync(file, Buffer.from(bytes));
  return file;
});

ipcMain.handle('edith:readBench', () => {
  const file = path.join(projectRoot(), 'output', 'bench.json');
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, 'utf-8'));
});
