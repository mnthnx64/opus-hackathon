'use client';

import type { NotationType } from '@/types/music';

interface NotationTypeSelectorProps {
  value: NotationType;
  onChange: (type: NotationType) => void;
}

const OPTIONS: { id: NotationType; label: string }[] = [
  { id: 'staff', label: 'Staff (Western)' },
  { id: 'hindustani', label: 'Hindustani (Sargam)' },
  { id: 'carnatic', label: 'Carnatic (Svara)' },
];

export default function NotationTypeSelector({ value, onChange }: NotationTypeSelectorProps) {
  return (
    <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
      {OPTIONS.map((opt) => (
        <button
          key={opt.id}
          onClick={() => onChange(opt.id)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            value === opt.id
              ? 'bg-indigo-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
