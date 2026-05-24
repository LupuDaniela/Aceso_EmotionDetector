export type NavView =
  | 'home'
  | 'conversation'
  | 'history'
  | 'stats'
  | 'calendar'
  | 'settings'
  | 'achievements'

export type MoodKey =
  | 'joy' | 'sadness' | 'fear' | 'anger'
  | 'surprise' | 'trust' | 'anticipation'
  | 'disgust' | 'love' | 'neutral'

export interface MoodConfig {
  emoji: string
  label: string
  color: string
}

export interface AchievementConfig {
  days:  number
  img:   string
  label: string
  name:  string
}

export interface CalendarProps {
  moodLog:      Record<string, MoodKey>
  currentYear:  number
  currentMonth: number
  canGoNext:    boolean
  onLogMood:    (dateKey: string, mood: MoodKey) => void
  onRemoveMood: (dateKey: string) => void
  onPrevMonth:  () => void
  onNextMonth:  () => void
  onToday:      () => void
}