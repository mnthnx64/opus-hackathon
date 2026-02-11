'use client';

import React from 'react';
import type { Note, Bar } from '@/types/music';
import { confidenceToColor } from '@/lib/noteUtils';

interface HindustaniRendererProps {
  notes: Note[];
  bars: Bar[];
  taalName?: string;
  highlightedIndex?: number;
}

function renderNote(
  note: Note,
  x: number,
  y: number,
  index: number,
  highlightedIndex: number,
): React.JSX.Element {
  const isHighlighted = index === highlightedIndex;
  const color = isHighlighted ? '#818cf8' : confidenceToColor(note.confidence);

  if (note.is_rest) {
    return (
      <g key={`note-${index}`} transform={`translate(${x}, ${y})`}>
        <text
          textAnchor="middle"
          fontSize="18"
          fill="#666"
          fontFamily="serif"
        >
          &ndash;
        </text>
      </g>
    );
  }

  const swara = note.swara || '?';
  const variant = note.swara_variant;
  const octaveOffset = note.octave_offset;

  return (
    <g key={`note-${index}`} transform={`translate(${x}, ${y})`}>
      {/* Highlight glow */}
      {isHighlighted && (
        <circle cx={0} cy={-4} r={16} fill="#818cf8" opacity={0.2} />
      )}

      {/* Swara text */}
      <text
        textAnchor="middle"
        fontSize="18"
        fontWeight="bold"
        fill={color}
        fontFamily="serif"
      >
        {swara}
      </text>

      {/* Komal indicator: underline */}
      {variant === 'komal' && (
        <line x1={-10} y1={4} x2={10} y2={4} stroke={color} strokeWidth={2} />
      )}

      {/* Tivra indicator: line above Ma */}
      {variant === 'tivra' && (
        <line x1={-10} y1={-18} x2={10} y2={-18} stroke={color} strokeWidth={2} />
      )}

      {/* Octave dots: above for taar (+1), below for mandra (-1) */}
      {octaveOffset > 0 &&
        Array.from({ length: octaveOffset }).map((_, i) => (
          <circle
            key={`dot-up-${i}`}
            cx={0}
            cy={-22 - i * 6}
            r={2}
            fill={color}
          />
        ))}
      {octaveOffset < 0 &&
        Array.from({ length: Math.abs(octaveOffset) }).map((_, i) => (
          <circle
            key={`dot-down-${i}`}
            cx={0}
            cy={10 + i * 6}
            r={2}
            fill={color}
          />
        ))}
    </g>
  );
}

export default function HindustaniRenderer({
  notes,
  bars,
  taalName = 'Teentaal',
  highlightedIndex = -1,
}: HindustaniRendererProps) {
  const noteSpacing = 48;
  const lineHeight = 80;
  const notesPerLine = 16; // Teentaal = 16 beats
  const xStart = 30;
  const yStart = 40;

  // Collect all rendered elements
  const elements: React.JSX.Element[] = [];

  const allNotes = bars.length > 0
    ? bars.flatMap((bar) => bar.beats.flatMap((beat) => beat.notes))
    : notes;

  const totalLines = Math.ceil(allNotes.length / notesPerLine);
  const svgWidth = notesPerLine * noteSpacing + xStart * 2;
  const svgHeight = totalLines * lineHeight + yStart * 2;

  // Draw taal markers and barlines
  for (let line = 0; line < totalLines; line++) {
    const y = yStart + line * lineHeight;

    // Horizontal baseline
    elements.push(
      <line
        key={`baseline-${line}`}
        x1={xStart - 10}
        y1={y + 10}
        x2={svgWidth - xStart + 10}
        y2={y + 10}
        stroke="#333"
        strokeWidth={0.5}
      />
    );

    // Taal division markers (every 4 beats for Teentaal)
    for (let div = 0; div <= notesPerLine; div += 4) {
      elements.push(
        <line
          key={`div-${line}-${div}`}
          x1={xStart + div * noteSpacing - 15}
          y1={y - 25}
          x2={xStart + div * noteSpacing - 15}
          y2={y + 15}
          stroke="#444"
          strokeWidth={1}
        />
      );
    }
  }

  // Render notes
  for (let i = 0; i < allNotes.length; i++) {
    const note = allNotes[i];
    const line = Math.floor(i / notesPerLine);
    const col = i % notesPerLine;
    const x = xStart + col * noteSpacing;
    const y = yStart + line * lineHeight;

    elements.push(renderNote(note, x, y, i, highlightedIndex));
  }

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-lg p-4">
      <div className="text-sm text-gray-400 mb-2 font-medium">
        {taalName} &middot; {allNotes.length} notes
      </div>
      <svg
        width={svgWidth}
        height={Math.max(svgHeight, 100)}
        className="min-h-[100px]"
      >
        {elements}
      </svg>
    </div>
  );
}
