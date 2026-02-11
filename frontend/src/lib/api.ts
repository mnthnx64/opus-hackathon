/** REST + WebSocket API helpers. */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

import type { TranscriptionResult, PitchOption, NotationTypeOption } from '@/types/music';

export async function transcribeAudio(
  file: File,
  basePitch: number = 60,
  notationType: string = 'staff',
): Promise<TranscriptionResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('base_pitch', String(basePitch));
  formData.append('notation_type', notationType);

  const res = await fetch(`${API_BASE}/api/transcribe`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Transcription failed');
  }

  return res.json();
}

export async function fetchPitches(): Promise<PitchOption[]> {
  const res = await fetch(`${API_BASE}/api/config/pitches`);
  return res.json();
}

export async function fetchNotationTypes(): Promise<NotationTypeOption[]> {
  const res = await fetch(`${API_BASE}/api/config/notation-types`);
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

export function createLiveWebSocket(): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/live`);
}

export { API_BASE, WS_BASE };
