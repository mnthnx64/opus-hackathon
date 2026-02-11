'use client';

import type { NotationType } from '@/types/music';
import PitchSelector from './PitchSelector';
import NotationTypeSelector from './NotationTypeSelector';

interface LiveCaptureProps {
  isListening: boolean;
  onStart: () => void;
  onStop: () => void;
  onFinalize: () => void;
  audioLevel: number;
  basePitch: number;
  onBasePitchChange: (pitch: number) => void;
  notationType: NotationType;
  onNotationTypeChange: (type: NotationType) => void;
  noteCount: number;
  error: string | null;
}

export default function LiveCapture({
  isListening,
  onStart,
  onStop,
  onFinalize,
  audioLevel,
  basePitch,
  onBasePitchChange,
  notationType,
  onNotationTypeChange,
  noteCount,
  error,
}: LiveCaptureProps) {
  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <PitchSelector value={basePitch} onChange={onBasePitchChange} />
        <NotationTypeSelector value={notationType} onChange={onNotationTypeChange} />
      </div>

      {/* Main button */}
      <div className="flex items-center gap-4">
        {!isListening ? (
          <button
            onClick={onStart}
            className="flex items-center gap-3 px-8 py-4 bg-red-600 hover:bg-red-700 text-white rounded-xl font-bold text-lg transition-all hover:scale-105"
          >
            <div className="w-4 h-4 rounded-full bg-white" />
            Start Listening
          </button>
        ) : (
          <>
            <button
              onClick={onStop}
              className="flex items-center gap-3 px-8 py-4 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-bold text-lg transition-colors"
            >
              <div className="w-4 h-4 rounded-sm bg-white" />
              Stop
            </button>
            <button
              onClick={onFinalize}
              className="px-6 py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold text-lg transition-colors"
            >
              Stop & Finalize
            </button>
          </>
        )}
      </div>

      {/* Status */}
      <div className="flex items-center gap-4">
        {isListening && (
          <>
            {/* Recording indicator */}
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-red-400 font-medium">Listening...</span>
            </div>

            {/* Audio level bar */}
            <div className="flex-1 max-w-xs h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full transition-all duration-100"
                style={{ width: `${Math.min(audioLevel * 500, 100)}%` }}
              />
            </div>
          </>
        )}

        <span className="text-sm text-gray-400">
          {noteCount} notes detected
        </span>
      </div>

      {error && (
        <div className="text-red-400 bg-red-900/20 border border-red-800 rounded-lg p-3 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
