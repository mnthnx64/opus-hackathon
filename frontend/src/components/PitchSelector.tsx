'use client';

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

interface PitchSelectorProps {
  value: number;
  onChange: (midi: number) => void;
}

const PITCH_OPTIONS: { midi: number; label: string }[] = [];
for (let octave = 2; octave <= 6; octave++) {
  for (let i = 0; i < NOTE_NAMES.length; i++) {
    const midi = (octave + 1) * 12 + i;
    PITCH_OPTIONS.push({ midi, label: `${NOTE_NAMES[i]}${octave}` });
  }
}

export default function PitchSelector({ value, onChange }: PitchSelectorProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
    >
      {PITCH_OPTIONS.map((opt) => (
        <option key={opt.midi} value={opt.midi}>
          {opt.label} (Sa)
        </option>
      ))}
    </select>
  );
}
