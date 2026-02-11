'use client';

import { useCallback } from 'react';
import SimpleEditor from '@/components/SimpleEditor';
import NotationDisplay from '@/components/NotationDisplay';
import PitchSelector from '@/components/PitchSelector';
import PlaybackControls from '@/components/PlaybackControls';
import { useEditorStore } from '@/stores/editorStore';
import { useTranscriptionStore } from '@/stores/transcriptionStore';
import { useLiveStore } from '@/stores/liveStore';
import { usePlayback } from '@/hooks/usePlayback';
import type { TranscriptionResult, Duration } from '@/types/music';

const DURATIONS: { id: Duration; label: string }[] = [
  { id: 'whole', label: 'Whole' },
  { id: 'half', label: 'Half' },
  { id: 'quarter', label: 'Quarter' },
  { id: 'eighth', label: 'Eighth' },
  { id: 'sixteenth', label: '16th' },
];

export default function EditorPage() {
  const {
    notes,
    notationType,
    basePitch,
    selectedDuration,
    tempo,
    addNote,
    removeNote,
    clearAll,
    setNotationType,
    setBasePitch,
    setSelectedDuration,
    setTempo,
    loadFromTranscription,
  } = useEditorStore();

  const transcriptionResult = useTranscriptionStore((s) => s.result);
  const liveResult = useLiveStore((s) => s.finalResult);

  const { isPlaying, currentNoteIndex, playSequence, playNote, stop } = usePlayback();

  const handlePlay = useCallback(() => {
    if (notes.length > 0) {
      playSequence(notes, tempo);
    }
  }, [notes, tempo, playSequence]);

  const handlePreviewNote = useCallback(
    (midiPitch: number) => {
      playNote(midiPitch, 0.3);
    },
    [playNote],
  );

  const handleLoadTranscription = useCallback(() => {
    if (transcriptionResult?.notes) {
      loadFromTranscription(transcriptionResult.notes);
    }
  }, [transcriptionResult, loadFromTranscription]);

  const handleLoadLive = useCallback(() => {
    if (liveResult?.notes) {
      loadFromTranscription(liveResult.notes);
    }
  }, [liveResult, loadFromTranscription]);

  const editorResult: TranscriptionResult = {
    notes,
    bars: [],
    notation_type: notationType,
    tempo,
    time_signature: '4/4',
    base_pitch: basePitch,
    base_pitch_name: '',
    taal_name: notationType === 'hindustani' ? 'Teentaal' : '',
    tala_name: notationType === 'carnatic' ? 'Adi' : '',
  };

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
    pdf.text('Sangeet - Editor', 14, 15);
    pdf.addImage(imgData, 'PNG', 10, 30, pdfWidth - 20, pdfHeight - 20);
    pdf.save('editor-notation.pdf');
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Notation Editor</h1>
        <p className="text-gray-400 mt-1">
          Click to place notes on the grid, or load from a transcription
        </p>
      </div>

      {/* Settings bar */}
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Base Pitch</label>
          <PitchSelector value={basePitch} onChange={setBasePitch} />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Duration</label>
          <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
            {DURATIONS.map((d) => (
              <button
                key={d.id}
                onClick={() => setSelectedDuration(d.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  selectedDuration === d.id
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Tempo</label>
          <input
            type="number"
            value={tempo}
            onChange={(e) => setTempo(Number(e.target.value))}
            min={40}
            max={300}
            className="w-20 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm"
          />
        </div>
      </div>

      {/* Load buttons */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleLoadTranscription}
          disabled={!transcriptionResult}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
        >
          Load from Upload
        </button>
        <button
          onClick={handleLoadLive}
          disabled={!liveResult}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
        >
          Load from Live
        </button>
        <button
          onClick={clearAll}
          disabled={notes.length === 0}
          className="px-4 py-2 bg-red-900/50 hover:bg-red-900 disabled:bg-gray-800 disabled:text-gray-600 text-red-300 rounded-lg text-sm font-medium transition-colors"
        >
          Clear All
        </button>
      </div>

      {/* Editor grid */}
      <SimpleEditor
        notes={notes}
        notationType={notationType}
        basePitch={basePitch}
        selectedDuration={selectedDuration}
        onAddNote={addNote}
        onRemoveNote={removeNote}
        onPreviewNote={handlePreviewNote}
      />

      {/* Notation preview + playback */}
      {notes.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <PlaybackControls
              isPlaying={isPlaying}
              onPlay={handlePlay}
              onStop={stop}
              tempo={tempo}
            />
            <button
              onClick={handleExportPDF}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Export PDF
            </button>
          </div>

          <NotationDisplay
            result={editorResult}
            notationType={notationType}
            onNotationTypeChange={setNotationType}
            highlightedIndex={currentNoteIndex}
          />
        </div>
      )}
    </div>
  );
}
