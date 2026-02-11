'use client';

import { useEffect, useRef, useState } from 'react';
import type { Note, Bar } from '@/types/music';
import { confidenceToColor } from '@/lib/noteUtils';

interface StaffRendererProps {
  notes: Note[];
  bars: Bar[];
  timeSignature?: string;
  highlightedIndex?: number;
}

function durationToVex(duration: string): string {
  switch (duration) {
    case 'whole': return 'w';
    case 'half': return 'h';
    case 'quarter': return 'q';
    case 'eighth': return '8';
    case 'sixteenth': return '16';
    default: return 'q';
  }
}

function midiToVexKey(midi: number): string {
  const noteNames = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
  const octave = Math.floor(midi / 12) - 1;
  const name = noteNames[midi % 12];
  return `${name}/${octave}`;
}

export default function StaffRenderer({
  notes,
  bars,
  timeSignature = '4/4',
  highlightedIndex = -1,
}: StaffRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || notes.length === 0) return;

    let cancelled = false;

    // Dynamic import to avoid Terser unicode issues at build time
    import('vexflow').then((VF) => {
      if (cancelled || !containerRef.current) return;

      const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental } = VF;

      containerRef.current.innerHTML = '';

      try {
        const renderer = new Renderer(containerRef.current, Renderer.Backends.SVG);
        const staveWidth = 300;
        const stavesPerLine = 3;
        const lineHeight = 120;
        const xStart = 20;
        const yStart = 20;

        const noteGroups: Note[][] = [];
        if (bars.length > 0) {
          for (const bar of bars) {
            const barNotes: Note[] = [];
            for (const beat of bar.beats) {
              barNotes.push(...beat.notes);
            }
            if (barNotes.length > 0) noteGroups.push(barNotes);
          }
        } else {
          for (let i = 0; i < notes.length; i += 4) {
            noteGroups.push(notes.slice(i, i + 4));
          }
        }

        const totalWidth = Math.min(noteGroups.length, stavesPerLine) * staveWidth + xStart * 2;
        const totalLines = Math.ceil(noteGroups.length / stavesPerLine);
        const totalHeight = totalLines * lineHeight + yStart * 2;
        renderer.resize(totalWidth, totalHeight);
        const context = renderer.getContext();

        let globalNoteIdx = 0;

        noteGroups.forEach((group, groupIdx) => {
          const lineNum = Math.floor(groupIdx / stavesPerLine);
          const colNum = groupIdx % stavesPerLine;
          const x = xStart + colNum * staveWidth;
          const y = yStart + lineNum * lineHeight;

          const stave = new Stave(x, y, staveWidth - 10);
          if (colNum === 0) {
            stave.addClef('treble');
            if (groupIdx === 0) {
              stave.addTimeSignature(timeSignature);
            }
          }
          stave.setContext(context).draw();

          const vexNotes: InstanceType<typeof StaveNote>[] = [];
          for (const note of group) {
            const isHighlighted = globalNoteIdx === highlightedIndex;
            globalNoteIdx++;

            if (note.is_rest) {
              const vn = new StaveNote({
                keys: ['b/4'],
                duration: durationToVex(note.duration) + 'r',
              });
              vexNotes.push(vn);
              continue;
            }

            const key = midiToVexKey(note.midi_pitch);
            const vn = new StaveNote({
              keys: [key],
              duration: durationToVex(note.duration),
            });

            if (note.accidental === '#') {
              vn.addModifier(new Accidental('#'));
            } else if (note.accidental === 'b') {
              vn.addModifier(new Accidental('b'));
            }

            const color = isHighlighted ? '#818cf8' : confidenceToColor(note.confidence);
            vn.setStyle({ fillStyle: color, strokeStyle: color });

            vexNotes.push(vn);
          }

          if (vexNotes.length > 0) {
            try {
              const voice = new Voice({ num_beats: 4, beat_value: 4 }).setMode(Voice.Mode.SOFT);
              voice.addTickables(vexNotes);
              new Formatter().joinVoices([voice]).format([voice], staveWidth - 60);
              voice.draw(context, stave);
            } catch {
              vexNotes.forEach((vn) => {
                try {
                  vn.setStave(stave);
                  vn.setContext(context);
                  vn.draw();
                } catch { /* skip broken notes */ }
              });
            }
          }
        });

        setError(null);
      } catch (err) {
        console.error('VexFlow render error:', err);
        setError(String(err));
      }
    }).catch((err) => {
      console.error('Failed to load VexFlow:', err);
      setError('Failed to load VexFlow library');
    });

    return () => { cancelled = true; };
  }, [notes, bars, timeSignature, highlightedIndex]);

  if (error) {
    return <div className="text-red-400 bg-red-900/20 rounded-lg p-4">{error}</div>;
  }

  return (
    <div
      ref={containerRef}
      className="overflow-x-auto bg-white rounded-lg p-4 min-h-[200px]"
    />
  );
}
