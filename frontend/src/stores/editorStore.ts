import { create } from 'zustand';
import type { Note, NotationType, Duration } from '@/types/music';

interface EditorState {
  notes: Note[];
  notationType: NotationType;
  basePitch: number;
  selectedDuration: Duration;
  tempo: number;
  addNote: (note: Note) => void;
  removeNote: (index: number) => void;
  clearAll: () => void;
  setNotationType: (type: NotationType) => void;
  setBasePitch: (pitch: number) => void;
  setSelectedDuration: (d: Duration) => void;
  setTempo: (tempo: number) => void;
  loadFromTranscription: (notes: Note[]) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  notes: [],
  notationType: 'staff',
  basePitch: 60,
  selectedDuration: 'quarter',
  tempo: 120,
  addNote: (note) =>
    set((state) => ({ notes: [...state.notes, note] })),
  removeNote: (index) =>
    set((state) => ({
      notes: state.notes.filter((_, i) => i !== index),
    })),
  clearAll: () => set({ notes: [] }),
  setNotationType: (notationType) => set({ notationType }),
  setBasePitch: (basePitch) => set({ basePitch }),
  setSelectedDuration: (selectedDuration) => set({ selectedDuration }),
  setTempo: (tempo) => set({ tempo }),
  loadFromTranscription: (notes) => set({ notes }),
}));
