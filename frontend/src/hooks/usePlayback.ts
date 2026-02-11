'use client';

import { useCallback, useRef, useState } from 'react';
import type { Note } from '@/types/music';
import { PlaybackEngine } from '@/audio/PlaybackEngine';

export function usePlayback() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentNoteIndex, setCurrentNoteIndex] = useState(-1);
  const engineRef = useRef<PlaybackEngine | null>(null);

  const getEngine = useCallback(() => {
    if (!engineRef.current) {
      engineRef.current = new PlaybackEngine();
    }
    return engineRef.current;
  }, []);

  const playSequence = useCallback(
    async (notes: Note[], tempo: number = 120) => {
      const engine = getEngine();
      await engine.init();
      setIsPlaying(true);
      setCurrentNoteIndex(0);

      await engine.playSequence(notes, tempo, (index) => {
        setCurrentNoteIndex(index);
      });

      setIsPlaying(false);
      setCurrentNoteIndex(-1);
    },
    [getEngine],
  );

  const playNote = useCallback(
    async (midiPitch: number, duration: number = 0.3) => {
      const engine = getEngine();
      await engine.init();
      engine.playNote(midiPitch, duration);
    },
    [getEngine],
  );

  const stop = useCallback(() => {
    engineRef.current?.stop();
    setIsPlaying(false);
    setCurrentNoteIndex(-1);
  }, []);

  return { isPlaying, currentNoteIndex, playSequence, playNote, stop };
}
