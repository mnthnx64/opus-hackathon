/** MIDI ↔ name conversions and notation helpers. */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function midiToNoteName(midi: number): string {
  if (midi < 0 || midi > 127) return 'Rest';
  const octave = Math.floor(midi / 12) - 1;
  const name = NOTE_NAMES[midi % 12];
  return `${name}${octave}`;
}

export function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function noteNameToMidi(name: string): number {
  const match = name.match(/^([A-G]#?)(\d+)$/);
  if (!match) return -1;
  const [, noteName, octaveStr] = match;
  const noteIndex = NOTE_NAMES.indexOf(noteName);
  if (noteIndex === -1) return -1;
  const octave = parseInt(octaveStr);
  return (octave + 1) * 12 + noteIndex;
}

export function durationToBeats(duration: string): number {
  switch (duration) {
    case 'whole': return 4;
    case 'half': return 2;
    case 'quarter': return 1;
    case 'eighth': return 0.5;
    case 'sixteenth': return 0.25;
    default: return 1;
  }
}

export function durationToSeconds(duration: string, tempo: number): number {
  const beats = durationToBeats(duration);
  return (beats * 60) / tempo;
}

export function confidenceToColor(confidence: number): string {
  if (confidence >= 0.8) return '#22c55e'; // green
  if (confidence >= 0.6) return '#eab308'; // yellow
  if (confidence >= 0.4) return '#f97316'; // orange
  return '#ef4444'; // red
}

export { NOTE_NAMES };
