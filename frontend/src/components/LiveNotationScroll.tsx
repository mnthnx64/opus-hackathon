'use client';

import { useEffect, useRef } from 'react';
import type { Note, NotationType } from '@/types/music';
import HindustaniRenderer from './HindustaniRenderer';
import CarnaticRenderer from './CarnaticRenderer';
import StaffRenderer from './StaffRenderer';

interface LiveNotationScrollProps {
  notes: Note[];
  notationType: NotationType;
}

export default function LiveNotationScroll({
  notes,
  notationType,
}: LiveNotationScrollProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll right as new notes arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [notes]);

  if (notes.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-800/50 rounded-xl border border-gray-700">
        <p className="text-gray-500 text-lg">
          Notes will appear here as you sing...
        </p>
      </div>
    );
  }

  // Highlight the last 3 notes
  const highlightIndex = notes.length - 1;

  return (
    <div
      ref={scrollRef}
      className="overflow-x-auto rounded-xl border border-gray-700"
    >
      {notationType === 'hindustani' && (
        <HindustaniRenderer
          notes={notes}
          bars={[]}
          taalName="Live"
          highlightedIndex={highlightIndex}
        />
      )}
      {notationType === 'carnatic' && (
        <CarnaticRenderer
          notes={notes}
          bars={[]}
          talaName="Live"
          highlightedIndex={highlightIndex}
        />
      )}
      {notationType === 'staff' && (
        <StaffRenderer
          notes={notes}
          bars={[]}
          highlightedIndex={highlightIndex}
        />
      )}

      {/* Glow effect on latest notes */}
      <style jsx>{`
        div :global(g:last-of-type text) {
          filter: drop-shadow(0 0 6px rgba(129, 140, 248, 0.6));
        }
      `}</style>
    </div>
  );
}
