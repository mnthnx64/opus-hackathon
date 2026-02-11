import { create } from 'zustand';
import type { TranscriptionResult, NotationType } from '@/types/music';

interface TranscriptionState {
  result: TranscriptionResult | null;
  isLoading: boolean;
  error: string | null;
  basePitch: number;
  notationType: NotationType;
  setResult: (result: TranscriptionResult | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setBasePitch: (pitch: number) => void;
  setNotationType: (type: NotationType) => void;
  reset: () => void;
}

export const useTranscriptionStore = create<TranscriptionState>((set) => ({
  result: null,
  isLoading: false,
  error: null,
  basePitch: 60,
  notationType: 'staff',
  setResult: (result) => set({ result, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  setBasePitch: (basePitch) => set({ basePitch }),
  setNotationType: (notationType) => set({ notationType }),
  reset: () => set({ result: null, isLoading: false, error: null }),
}));
