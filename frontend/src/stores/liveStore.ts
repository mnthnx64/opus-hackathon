import { create } from 'zustand';
import type { Note, NotationType, TranscriptionResult } from '@/types/music';

interface LiveState {
  isListening: boolean;
  notes: Note[];
  basePitch: number;
  notationType: NotationType;
  error: string | null;
  finalResult: TranscriptionResult | null;
  setListening: (listening: boolean) => void;
  addNotes: (newNotes: Note[]) => void;
  setBasePitch: (pitch: number) => void;
  setNotationType: (type: NotationType) => void;
  setError: (error: string | null) => void;
  finalize: () => void;
  reset: () => void;
}

export const useLiveStore = create<LiveState>((set, get) => ({
  isListening: false,
  notes: [],
  basePitch: 60,
  notationType: 'hindustani',
  error: null,
  finalResult: null,
  setListening: (isListening) => set({ isListening }),
  addNotes: (newNotes) =>
    set((state) => ({ notes: [...state.notes, ...newNotes] })),
  setBasePitch: (basePitch) => set({ basePitch }),
  setNotationType: (notationType) => set({ notationType }),
  setError: (error) => set({ error }),
  finalize: () => {
    const state = get();
    set({
      isListening: false,
      finalResult: {
        notes: state.notes,
        bars: [],
        notation_type: state.notationType,
        tempo: 120,
        time_signature: '4/4',
        base_pitch: state.basePitch,
        base_pitch_name: '',
        taal_name: state.notationType === 'hindustani' ? 'Teentaal' : '',
        tala_name: state.notationType === 'carnatic' ? 'Adi' : '',
      },
    });
  },
  reset: () =>
    set({
      isListening: false,
      notes: [],
      error: null,
      finalResult: null,
    }),
}));
