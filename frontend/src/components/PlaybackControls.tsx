'use client';

interface PlaybackControlsProps {
  isPlaying: boolean;
  onPlay: () => void;
  onStop: () => void;
  tempo?: number;
  disabled?: boolean;
}

export default function PlaybackControls({
  isPlaying,
  onPlay,
  onStop,
  tempo,
  disabled = false,
}: PlaybackControlsProps) {
  return (
    <div className="flex items-center gap-3">
      {!isPlaying ? (
        <button
          onClick={onPlay}
          disabled={disabled}
          className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
          </svg>
          Play
        </button>
      ) : (
        <button
          onClick={onStop}
          className="flex items-center gap-2 px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M5.75 3A1.75 1.75 0 004 4.75v10.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0016 15.25V4.75A1.75 1.75 0 0014.25 3h-8.5z" />
          </svg>
          Stop
        </button>
      )}
      {tempo && (
        <span className="text-sm text-gray-400">
          {Math.round(tempo)} BPM
        </span>
      )}
    </div>
  );
}
