/**
 * Tone.js PolySynth playback engine.
 *
 * Uses dynamic import to avoid Terser unicode minification issues at build time.
 * Uses triangle wave for a clean, instrument-like sound.
 */

import type { Note } from '@/types/music';
import { midiToFrequency, durationToSeconds } from '@/lib/noteUtils';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ToneModule: any = null;

async function loadTone() {
  if (!ToneModule) {
    ToneModule = await import('tone');
  }
  return ToneModule;
}

export class PlaybackEngine {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private synth: any = null;
  private isInitialized = false;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private currentPart: any = null;

  async init(): Promise<void> {
    if (this.isInitialized) return;
    const Tone = await loadTone();
    await Tone.start();
    this.synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle' },
      envelope: {
        attack: 0.02,
        decay: 0.1,
        sustain: 0.3,
        release: 0.3,
      },
    }).toDestination();
    this.isInitialized = true;
  }

  async playSequence(
    notes: Note[],
    tempo: number = 120,
    onNoteChange?: (index: number) => void,
  ): Promise<void> {
    if (!this.synth) await this.init();
    const Tone = await loadTone();

    this.stop();

    return new Promise<void>((resolve) => {
      const playableNotes = notes.filter((n) => !n.is_rest && n.midi_pitch > 0);
      if (playableNotes.length === 0) {
        resolve();
        return;
      }

      const startOffset = notes[0]?.start_time || 0;
      const events: Array<{ time: number; note: Note; index: number }> = [];

      notes.forEach((note, index) => {
        events.push({
          time: note.start_time - startOffset,
          note,
          index,
        });
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const part = new Tone.Part((time: number, event: any) => {
        if (!event.note.is_rest && event.note.midi_pitch > 0) {
          const freq = midiToFrequency(event.note.midi_pitch);
          const dur = durationToSeconds(event.note.duration, tempo);
          this.synth?.triggerAttackRelease(freq, dur, time);
        }
        Tone.getDraw().schedule(() => {
          onNoteChange?.(event.index);
        }, time);
      }, events);

      this.currentPart = part;

      const lastNote = notes[notes.length - 1];
      const totalDuration =
        lastNote.end_time - startOffset + durationToSeconds(lastNote.duration, tempo);

      part.start(0);
      Tone.getTransport().start();

      setTimeout(() => {
        this.stop();
        resolve();
      }, totalDuration * 1000 + 500);
    });
  }

  async playNote(midiPitch: number, duration: number = 0.3): Promise<void> {
    if (!this.synth) await this.init();
    const freq = midiToFrequency(midiPitch);
    this.synth.triggerAttackRelease(freq, duration);
  }

  async stop(): Promise<void> {
    const Tone = ToneModule;
    if (!Tone) return;
    this.currentPart?.stop();
    this.currentPart?.dispose();
    this.currentPart = null;
    try {
      Tone.getTransport().stop();
      Tone.getTransport().position = 0;
    } catch { /* ignore if transport not started */ }
  }
}
