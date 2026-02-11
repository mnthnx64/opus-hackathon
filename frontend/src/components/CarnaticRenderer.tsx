'use client';

import React from 'react';
import type { Note, Bar } from '@/types/music';
import { confidenceToColor } from '@/lib/noteUtils';

interface CarnaticRendererProps {
  notes: Note[];
  bars: Bar[];
  talaName?: string;
  highlightedIndex?: number;
}

function renderCarnaticNote(
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
          ,
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

      {/* Variant number as subscript */}
      {variant && (
        <text
          textAnchor="start"
          x={12}
          y={6}
          fontSize="11"
          fill={color}
          fontFamily="serif"
        >
          {variant}
        </text>
      )}

      {/* Octave dots */}
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

export default function CarnaticRenderer({
  notes,
  bars,
  talaName = 'Adi',
  highlightedIndex = -1,
}: CarnaticRendererProps) {
  const noteSpacing = 52;
  const lineHeight = 80;
  const notesPerLine = 8; // Adi tala = 8 beats
  const xStart = 40;
  const yStart = 40;

  const elements: React.JSX.Element[] = [];

  const allNotes = bars.length > 0
    ? bars.flatMap((bar) => bar.beats.flatMap((beat) => beat.notes))
    : notes;

  const totalLines = Math.ceil(allNotes.length / notesPerLine);
  const svgWidth = notesPerLine * noteSpacing + xStart * 2;
  const svgHeight = totalLines * lineHeight + yStart * 2;

  // Draw tala markers (Adi: 4+2+2)
  const talaPattern = [4, 2, 2];

  for (let line = 0; line < totalLines; line++) {
    const y = yStart + line * lineHeight;

    // Baseline
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

    // Tala divisions
    let beatPos = 0;
    for (let d = 0; d <= talaPattern.length; d++) {
      elements.push(
        <line
          key={`div-${line}-${d}`}
          x1={xStart + beatPos * noteSpacing - 18}
          y1={y - 25}
          x2={xStart + beatPos * noteSpacing - 18}
          y2={y + 15}
          stroke="#444"
          strokeWidth={d === 0 ? 2 : 1}
        />
      );
      // Tala symbol
      if (d === 0) {
        elements.push(
          <text
            key={`sym-${line}-${d}`}
            x={xStart + beatPos * noteSpacing - 18}
            y={y - 28}
            textAnchor="middle"
            fontSize="10"
            fill="#888"
          >
            ||
          </text>
        );
      }
      if (d < talaPattern.length) beatPos += talaPattern[d];
    }
  }

  // Render notes
  for (let i = 0; i < allNotes.length; i++) {
    const note = allNotes[i];
    const line = Math.floor(i / notesPerLine);
    const col = i % notesPerLine;
    const x = xStart + col * noteSpacing;
    const y = yStart + line * lineHeight;

    elements.push(renderCarnaticNote(note, x, y, i, highlightedIndex));
  }

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-lg p-4">
      <div className="text-sm text-gray-400 mb-2 font-medium">
        {talaName} Tala &middot; {allNotes.length} notes
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
