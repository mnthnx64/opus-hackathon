'use client';

import type { TranscriptionResult, NotationType } from '@/types/music';
import StaffRenderer from './StaffRenderer';
import HindustaniRenderer from './HindustaniRenderer';
import CarnaticRenderer from './CarnaticRenderer';
import NotationTypeSelector from './NotationTypeSelector';

interface NotationDisplayProps {
  result: TranscriptionResult;
  notationType: NotationType;
  onNotationTypeChange: (type: NotationType) => void;
  highlightedIndex?: number;
}

export default function NotationDisplay({
  result,
  notationType,
  onNotationTypeChange,
  highlightedIndex = -1,
}: NotationDisplayProps) {
  return (
    <div className="space-y-4">
      {/* Metadata header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4 text-sm text-gray-400">
          <span>Tempo: {Math.round(result.tempo)} BPM</span>
          <span>Time: {result.time_signature}</span>
          {result.base_pitch_name && (
            <span>Base: {result.base_pitch_name}</span>
          )}
          {result.taal_name && <span>Taal: {result.taal_name}</span>}
          {result.tala_name && <span>Tala: {result.tala_name}</span>}
          <span>{result.notes.length} notes</span>
        </div>
        <NotationTypeSelector
          value={notationType}
          onChange={onNotationTypeChange}
        />
      </div>

      {/* Renderer */}
      <div id="notation-display">
        {notationType === 'staff' && (
          <StaffRenderer
            notes={result.notes}
            bars={result.bars}
            timeSignature={result.time_signature}
            highlightedIndex={highlightedIndex}
          />
        )}
        {notationType === 'hindustani' && (
          <HindustaniRenderer
            notes={result.notes}
            bars={result.bars}
            taalName={result.taal_name || 'Teentaal'}
            highlightedIndex={highlightedIndex}
          />
        )}
        {notationType === 'carnatic' && (
          <CarnaticRenderer
            notes={result.notes}
            bars={result.bars}
            talaName={result.tala_name || 'Adi'}
            highlightedIndex={highlightedIndex}
          />
        )}
      </div>

      {result.notes.length === 0 && (
        <div className="text-center text-gray-500 py-8">
          No notes detected. Try a different audio file or adjust settings.
        </div>
      )}
    </div>
  );
}
