'use client';

import { useCallback } from 'react';
import type { Note, Duration, NotationType } from '@/types/music';

interface SimpleEditorProps {
  notes: Note[];
  notationType: NotationType;
  basePitch: number;
  selectedDuration: Duration;
  onAddNote: (note: Note) => void;
  onRemoveNote: (index: number) => void;
  onPreviewNote: (midiPitch: number) => void;
}

const DURATIONS: { id: Duration; label: string; symbol: string }[] = [
  { id: 'whole', label: 'Whole', symbol: 'W' },
  { id: 'half', label: 'Half', symbol: 'H' },
  { id: 'quarter', label: 'Quarter', symbol: 'Q' },
  { id: 'eighth', label: 'Eighth', symbol: '8' },
  { id: 'sixteenth', label: '16th', symbol: '16' },
];

// Define rows based on notation type
function getRows(notationType: NotationType, basePitch: number) {
  if (notationType === 'staff') {
    // Two octaves of chromatic pitches
    const rows = [];
    for (let midi = basePitch + 24; midi >= basePitch - 12; midi--) {
      const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      const octave = Math.floor(midi / 12) - 1;
      const name = noteNames[midi % 12];
      rows.push({ midi, label: `${name}${octave}` });
    }
    return rows;
  }

  // Indian notation: swaras across 3 octaves
  const swaras = notationType === 'hindustani'
    ? ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni']
    : ['Sa', 'Ri', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni'];

  const semitoneMaps = notationType === 'hindustani'
    ? [0, 2, 4, 5, 7, 9, 11]
    : [0, 2, 4, 5, 7, 9, 11];

  const rows = [];
  for (let oct = 1; oct >= -1; oct--) {
    for (let i = swaras.length - 1; i >= 0; i--) {
      const midi = basePitch + oct * 12 + semitoneMaps[i];
      const prefix = oct > 0 ? "'" : oct < 0 ? '.' : '';
      rows.push({ midi, label: `${prefix}${swaras[i]}${prefix}` });
    }
  }
  return rows;
}

export default function SimpleEditor({
  notes,
  notationType,
  basePitch,
  selectedDuration,
  onAddNote,
  onRemoveNote,
  onPreviewNote,
}: SimpleEditorProps) {
  const rows = getRows(notationType, basePitch);
  const numCols = Math.max(notes.length + 8, 16); // Extra empty columns

  const handleCellClick = useCallback(
    (midi: number, colIndex: number) => {
      // Check if note already exists at this position
      const existingIdx = notes.findIndex(
        (n, i) => i === colIndex && n.midi_pitch === midi,
      );
      if (existingIdx !== -1) {
        onRemoveNote(existingIdx);
        return;
      }

      // Add note
      const startTime = colIndex * 0.5;
      const note: Note = {
        midi_pitch: midi,
        start_time: startTime,
        end_time: startTime + 0.5,
        confidence: 1.0,
        duration: selectedDuration,
        is_rest: false,
        note_name: '',
        accidental: '',
        octave: Math.floor(midi / 12) - 1,
        swara: '',
        swara_variant: '',
        octave_offset: 0,
      };
      onAddNote(note);
      onPreviewNote(midi);
    },
    [notes, selectedDuration, onAddNote, onRemoveNote, onPreviewNote],
  );

  return (
    <div className="space-y-3">
      {/* Duration selector toolbar */}
      <div className="flex gap-2">
        {DURATIONS.map((d) => (
          <button
            key={d.id}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              selectedDuration === d.id
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
            onClick={() => {
              // This calls the parent's setSelectedDuration via EditorPage
            }}
            title={d.label}
          >
            {d.label}
          </button>
        ))}
      </div>

      {/* Grid */}
      <div className="overflow-auto max-h-[500px] border border-gray-700 rounded-lg">
        <div className="inline-block min-w-full">
          <div className="grid" style={{ gridTemplateColumns: `80px repeat(${numCols}, 36px)` }}>
            {/* Header row */}
            <div className="sticky left-0 z-10 bg-gray-800 border-b border-gray-600 px-2 py-1" />
            {Array.from({ length: numCols }).map((_, col) => (
              <div
                key={`hdr-${col}`}
                className="border-b border-gray-700 px-1 py-1 text-center text-xs text-gray-500"
              >
                {col + 1}
              </div>
            ))}

            {/* Pitch rows */}
            {rows.map((row) => (
              <>
                <div
                  key={`label-${row.midi}`}
                  className="sticky left-0 z-10 bg-gray-800 border-b border-gray-700 px-2 py-0.5 text-xs text-gray-300 font-mono flex items-center"
                >
                  {row.label}
                </div>
                {Array.from({ length: numCols }).map((_, col) => {
                  const isActive = notes.some(
                    (n, i) => i === col && n.midi_pitch === row.midi,
                  );
                  return (
                    <div
                      key={`cell-${row.midi}-${col}`}
                      onClick={() => handleCellClick(row.midi, col)}
                      className={`
                        border-b border-r border-gray-800 cursor-pointer transition-colors h-6
                        ${isActive
                          ? 'bg-indigo-500 hover:bg-indigo-400'
                          : 'bg-gray-900 hover:bg-gray-700'
                        }
                        ${col % 4 === 0 ? 'border-l border-l-gray-600' : ''}
                      `}
                    />
                  );
                })}
              </>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
