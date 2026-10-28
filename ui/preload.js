'use strict';

// Preload: the only API surface the renderer can see. No Node/Electron objects
// leak through; only the allow-listed functions below.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('edith', {
  ping: () => ipcRenderer.invoke('edith:ping'),
  transcribe: (payload) => ipcRenderer.invoke('edith:transcribe', payload),
  pickAudioFile: () => ipcRenderer.invoke('edith:pickAudioFile'),
  saveRecording: (bytes) => ipcRenderer.invoke('edith:saveRecording', bytes),
  readBench: () => ipcRenderer.invoke('edith:readBench'),
});
