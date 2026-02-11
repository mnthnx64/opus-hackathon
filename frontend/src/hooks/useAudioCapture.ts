'use client';

import { useCallback, useRef, useState } from 'react';

interface UseAudioCaptureOptions {
  sampleRate?: number;
  bufferSize?: number;
  onAudioData?: (data: Float32Array) => void;
}

export function useAudioCapture({
  sampleRate = 22050,
  bufferSize = 4096,
  onAudioData,
}: UseAudioCaptureOptions = {}) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);

  const start = useCallback(async () => {
    try {
      setError(null);

      // Request mic
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // Create AudioContext
      const ctx = new AudioContext({ sampleRate });
      contextRef.current = ctx;

      // Load AudioWorklet
      await ctx.audioWorklet.addModule('/audio-processor.js');
      const worklet = new AudioWorkletNode(ctx, 'audio-processor', {
        processorOptions: { bufferSize },
      });
      workletRef.current = worklet;

      // Handle audio data from worklet
      worklet.port.onmessage = (event) => {
        const { audioData, level } = event.data;
        if (audioData) {
          const float32 = new Float32Array(audioData);
          setAudioLevel(level || 0);
          onAudioData?.(float32);
        }
      };

      // Connect: mic → worklet
      const source = ctx.createMediaStreamSource(stream);
      source.connect(worklet);
      // Don't connect worklet to destination (no feedback)

      setIsCapturing(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to capture audio';
      setError(message);
      console.error('Audio capture error:', err);
    }
  }, [sampleRate, bufferSize, onAudioData]);

  const stop = useCallback(() => {
    workletRef.current?.disconnect();
    workletRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    contextRef.current?.close();
    contextRef.current = null;

    setIsCapturing(false);
    setAudioLevel(0);
  }, []);

  return { isCapturing, audioLevel, error, start, stop };
}
