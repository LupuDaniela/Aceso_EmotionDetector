import type { MoodKey, MoodConfig } from '@/types'

export const MOOD_CONFIG: Record<MoodKey, MoodConfig> = {
  joy:          { emoji: '😄', label: 'Bucurie',    color: 'rgba(72,199,142,0.80)'  },
  sadness:      { emoji: '😢', label: 'Tristețe',   color: 'rgba(80,148,215,0.80)'  },
  fear:         { emoji: '😱', label: 'Frică',      color: 'rgba(158,119,248,0.80)' },
  anger:        { emoji: '😡', label: 'Furie',      color: 'rgba(215,80,65,0.80)'   },
  surprise:     { emoji: '😯', label: 'Surpriză',   color: 'rgba(248,196,54,0.80)'  },
  trust:        { emoji: '🤝', label: 'Încredere',  color: 'rgba(50,188,198,0.80)'  },
  anticipation: { emoji: '🤩', label: 'Anticipare', color: 'rgba(253,150,38,0.80)'  },
  disgust:      { emoji: '🤢', label: 'Dezgust',    color: 'rgba(108,168,66,0.80)'  },
  love:         { emoji: '🥰', label: 'Iubire',     color: 'rgba(252,118,152,0.80)' },
  neutral:      { emoji: '😐', label: 'Neutru',     color: 'rgba(168,160,200,0.80)' },
}

export const MONTHS_RO = [
  'Ianuarie', 'Februarie', 'Martie',    'Aprilie',
  'Mai',      'Iunie',     'Iulie',     'August',
  'Septembrie','Octombrie', 'Noiembrie', 'Decembrie',
]

export const WEEKDAYS_SHORT = ['Lu', 'Ma', 'Mi', 'Jo', 'Vi', 'Sâ', 'Du']