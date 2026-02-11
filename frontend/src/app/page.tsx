'use client';

import { useCallback, useRef } from 'react';
import AudioUploader from '@/components/AudioUploader';
import PitchSelector from '@/components/PitchSelector';
import NotationTypeSelector from '@/components/NotationTypeSelector';
import NotationDisplay from '@/components/NotationDisplay';
import PlaybackControls from '@/components/PlaybackControls';
import { useTranscriptionStore } from '@/stores/transcriptionStore';
import { usePlayback } from '@/hooks/usePlayback';
import { transcribeAudio } from '@/lib/api';
import type { NotationType } from '@/types/music';

export default function UploadPage() {
  const {
    result,
    isLoading,
    error,
    basePitch,
    notationType,
    setResult,
    setLoading,
    setError,
    setBasePitch,
    setNotationType,
  } = useTranscriptionStore();

  const { isPlaying, currentNoteIndex, playSequence, stop } = usePlayback();
  const displayNotationType = useRef<NotationType>(notationType);

  const handleFileSelected = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const data = await transcribeAudio(file, basePitch, notationType);
        setResult(data);
        displayNotationType.current = notationType;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Transcription failed';
        setError(message);
      }
    },
    [basePitch, notationType, setResult, setLoading, setError],
  );

  const handlePlay = useCallback(() => {
    if (result && result.notes.length > 0) {
      playSequence(result.notes, result.tempo);
    }
  }, [result, playSequence]);

  const handleNotationTypeChange = useCallback(
    (type: NotationType) => {
      displayNotationType.current = type;
      setNotationType(type);
    },
    [setNotationType],
  );

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
    pdf.text('Sangeet - Musical Notation', 14, 15);
    pdf.setFontSize(10);
    pdf.text(
      `Tempo: ${Math.round(result?.tempo || 120)} BPM | Base: ${result?.base_pitch_name || 'C'} | Type: ${displayNotationType.current}`,
      14,
      22,
    );

    pdf.addImage(imgData, 'PNG', 10, 30, pdfWidth - 20, pdfHeight - 20);
    pdf.save('notation.pdf');
  }, [result]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Upload Audio</h1>
        <p className="text-gray-400 mt-1">
          Upload a music file to transcribe it into notation
        </p>
      </div>

      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Base Pitch (Sa)</label>
          <PitchSelector value={basePitch} onChange={setBasePitch} />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Notation Type</label>
          <NotationTypeSelector value={notationType} onChange={setNotationType} />
        </div>
      </div>

      <AudioUploader onFileSelected={handleFileSelected} isLoading={isLoading} />

      {error && (
        <div className="text-red-400 bg-red-900/20 border border-red-800 rounded-lg p-4">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <PlaybackControls
              isPlaying={isPlaying}
              onPlay={handlePlay}
              onStop={stop}
              tempo={result.tempo}
              disabled={result.notes.length === 0}
            />
            <button
              onClick={handleExportPDF}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Export PDF
            </button>
          </div>

          <NotationDisplay
            result={result}
            notationType={displayNotationType.current}
            onNotationTypeChange={handleNotationTypeChange}
            highlightedIndex={currentNoteIndex}
          />
        </div>
      )}
    </div>
  );
}
