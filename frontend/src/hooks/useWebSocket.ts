'use client';

import { useCallback, useRef, useState, useEffect } from 'react';
import type { TranscriptionResult } from '@/types/music';

interface UseWebSocketOptions {
  url: string;
  onNotes?: (result: TranscriptionResult) => void;
  onError?: (error: string) => void;
}

export function useWebSocket({ url, onNotes, onError }: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onNotesRef = useRef(onNotes);
  const onErrorRef = useRef(onError);

  // Keep callbacks up to date
  useEffect(() => {
    onNotesRef.current = onNotes;
    onErrorRef.current = onError;
  }, [onNotes, onError]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'notes' && msg.data) {
          onNotesRef.current?.(msg.data);
        } else if (msg.type === 'error') {
          onErrorRef.current?.(msg.message);
        }
      } catch {
        console.error('Failed to parse WS message');
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = () => {
      onErrorRef.current?.('WebSocket connection error');
      setIsConnected(false);
    };
  }, [url]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setIsConnected(false);
  }, []);

  const sendAudio = useCallback((data: Float32Array) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data.buffer);
    }
  }, []);

  const sendConfig = useCallback(
    (config: { base_pitch?: number; notation_type?: string }) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'config', ...config }));
      }
    },
    [],
  );

  return { isConnected, connect, disconnect, sendAudio, sendConfig };
}
