import type { MoodKey, MoodConfig } from '@/types'

export const MOOD_CONFIG: Record<MoodKey, MoodConfig> = {
  joy:          { emoji: '😄', label: 'Bucurie',    color: 'rgba(253,216,53,0.80)'  },
  sadness:      { emoji: '😢', label: 'Tristețe',   color: 'rgba(2,52,93,0.80)'     },
  fear:         { emoji: '😱', label: 'Frică',      color: 'rgba(171,71,188,0.80)'  },
  anger:        { emoji: '😡', label: 'Furie',      color: 'rgba(239,83,80,0.80)'   },
  surprise:     { emoji: '😯', label: 'Surpriză',   color: 'rgba(255,112,67,0.80)'  },
  trust:        { emoji: '🤝', label: 'Încredere',  color: 'rgba(38,166,154,0.80)'  },
  anticipation: { emoji: '🤩', label: 'Anticipare', color: 'rgba(255,167,38,0.80)'  },
  disgust:      { emoji: '🤢', label: 'Dezgust',    color: 'rgba(102,187,106,0.80)' },
  neutral:      { emoji: '😐', label: 'Neutru',     color: 'rgba(120,104,66,0.80)'  },
}

export const MONTHS_RO = [
  'Ianuarie', 'Februarie', 'Martie',    'Aprilie',
  'Mai',      'Iunie',     'Iulie',     'August',
  'Septembrie','Octombrie', 'Noiembrie', 'Decembrie',
]

export const WEEKDAYS_SHORT = ['Lu', 'Ma', 'Mi', 'Jo', 'Vi', 'Sâ', 'Du']