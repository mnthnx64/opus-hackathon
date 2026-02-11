'use client';

import { useCallback } from 'react';
import LiveCapture from '@/components/LiveCapture';
import LiveNotationScroll from '@/components/LiveNotationScroll';
import NotationDisplay from '@/components/NotationDisplay';
import PlaybackControls from '@/components/PlaybackControls';
import { useLiveStore } from '@/stores/liveStore';
import { useAudioCapture } from '@/hooks/useAudioCapture';
import { useWebSocket } from '@/hooks/useWebSocket';
import { usePlayback } from '@/hooks/usePlayback';
import type { TranscriptionResult } from '@/types/music';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export default function LivePage() {
  const {
    isListening,
    notes,
    basePitch,
    notationType,
    error,
    finalResult,
    setListening,
    addNotes,
    setBasePitch,
    setNotationType,
    setError,
    finalize,
    reset,
  } = useLiveStore();

  const { isPlaying, currentNoteIndex, playSequence, stop: stopPlayback } = usePlayback();

  const handleNotes = useCallback(
    (result: TranscriptionResult) => {
      if (result.notes.length > 0) {
        addNotes(result.notes);
      }
    },
    [addNotes],
  );

  const {
    connect,
    disconnect,
    sendAudio,
    sendConfig,
  } = useWebSocket({
    url: `${WS_URL}/ws/live`,
    onNotes: handleNotes,
    onError: setError,
  });

  const { audioLevel, start: startCapture, stop: stopCapture } = useAudioCapture({
    sampleRate: 22050,
    bufferSize: 4096,
    onAudioData: sendAudio,
  });

  const handleStart = useCallback(async () => {
    reset();
    connect();
    setTimeout(() => {
      sendConfig({ base_pitch: basePitch, notation_type: notationType });
    }, 500);
    await startCapture();
    setListening(true);
  }, [reset, connect, sendConfig, basePitch, notationType, startCapture, setListening]);

  const handleStop = useCallback(() => {
    stopCapture();
    disconnect();
    setListening(false);
  }, [stopCapture, disconnect, setListening]);

  const handleFinalize = useCallback(() => {
    handleStop();
    finalize();
  }, [handleStop, finalize]);

  const handlePlay = useCallback(() => {
    const target = finalResult || { notes, tempo: 120 };
    if (target.notes.length > 0) {
      playSequence(target.notes, target.tempo || 120);
    }
  }, [finalResult, notes, playSequence]);

  const handleExportPDF = useCallback(async () => {
    const el = document.getElementById('notation-display');
    if (!el) return;

    const html2canvas = (await import('html2canvas')).default;
    const { jsPDF } = await import('jspdf');

    const canvas = await html2canvas(el, { backgroundColor: '#111827' });
    const imgData = canvas.toDataURL('image/png');

    const pdf = new jsPDF('landscape', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

    pdf.setFontSize(16);
    pdf.text('Sangeet - Live Transcription', 14, 15);
    pdf.addImage(imgData, 'PNG', 10, 30, pdfWidth - 20, pdfHeight - 20);
    pdf.save('live-notation.pdf');
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Live Mode</h1>
        <p className="text-gray-400 mt-1">
          Sing or play into your microphone — see notation appear in real-time
        </p>
      </div>

      <LiveCapture
        isListening={isListening}
        onStart={handleStart}
        onStop={handleStop}
        onFinalize={handleFinalize}
        audioLevel={audioLevel}
        basePitch={basePitch}
        onBasePitchChange={setBasePitch}
        notationType={notationType}
        onNotationTypeChange={setNotationType}
        noteCount={notes.length}
        error={error}
      />

      {(isListening || notes.length > 0) && !finalResult && (
        <LiveNotationScroll notes={notes} notationType={notationType} />
      )}

      {finalResult && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <PlaybackControls
                isPlaying={isPlaying}
                onPlay={handlePlay}
                onStop={stopPlayback}
                tempo={finalResult.tempo}
              />
              <a
                href="/editor"
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Send to Editor
              </a>
            </div>
            <button
              onClick={handleExportPDF}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Export PDF
            </button>
          </div>

          <NotationDisplay
            result={finalResult}
            notationType={notationType}
            onNotationTypeChange={setNotationType}
            highlightedIndex={currentNoteIndex}
          />
        </div>
      )}
    </div>
  );
}
