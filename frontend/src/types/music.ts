/** TypeScript interfaces mirroring the IMR (Intermediate Music Representation). */

export type NotationType = 'staff' | 'hindustani' | 'carnatic';

export type Duration = 'whole' | 'half' | 'quarter' | 'eighth' | 'sixteenth';

export interface Note {
  midi_pitch: number;
  start_time: number;
  end_time: number;
  confidence: number;
  duration: Duration;
  is_rest: boolean;
  // Staff fields
  note_name: string;
  accidental: string;
  octave: number;
  // Indian notation fields
  swara: string;
  swara_variant: string;
  octave_offset: number;
}

export interface Beat {
  beat_number: number;
  notes: Note[];
}

export interface Bar {
  bar_number: number;
  beats: Beat[];
  time_signature: string;
}

export interface TranscriptionResult {
  notes: Note[];
  bars: Bar[];
  notation_type: NotationType;
  tempo: number;
  time_signature: string;
  base_pitch: number;
  base_pitch_name: string;
  taal_name: string;
  tala_name: string;
}

export interface PitchOption {
  midi: number;
  name: string;
  label: string;
}

export interface NotationTypeOption {
  id: NotationType;
  name: string;
  description: string;
}
