import { useState, useMemo, useCallback } from 'react'
import { toDateKey, getDaysInMonth } from '@/utils/date'
import type { MoodKey, CalendarProps } from '@/types'

type MoodLog = Record<string, MoodKey>

function calculateStreak(moodLog: MoodLog): number {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const cursor = new Date(today)

  if (!moodLog[toDateKey(cursor)]) {
    cursor.setDate(cursor.getDate() - 1)
  }

  let streak = 0
  while (moodLog[toDateKey(cursor)]) {
    streak++
    cursor.setDate(cursor.getDate() - 1)
  }
  return streak
}

export interface UseCalendarReturn extends CalendarProps {
  streak: number
}

export function useCalendar(storageKey = 'aceso_mood_log'): UseCalendarReturn {
  const [moodLog, setMoodLog] = useState<MoodLog>(() => {
    try {
      const raw = localStorage.getItem(storageKey)
      return raw ? (JSON.parse(raw) as MoodLog) : {}
    } catch {
      return {}
    }
  })

  const [calView, setCalView] = useState(() => {
    const now = new Date()
    return { year: now.getFullYear(), month: now.getMonth() }
  })

  const streak = useMemo(() => calculateStreak(moodLog), [moodLog])

  const canGoNext = useMemo(() => {
    const now = new Date()
    return (
      calView.year < now.getFullYear() ||
      (calView.year === now.getFullYear() && calView.month < now.getMonth())
    )
  }, [calView])

  const onLogMood = useCallback((dateKey: string, mood: MoodKey) => {
    setMoodLog(prev => {
      const next = { ...prev, [dateKey]: mood }
      localStorage.setItem(storageKey, JSON.stringify(next))
      return next
    })
  }, [storageKey])

  const onRemoveMood = useCallback((dateKey: string) => {
    setMoodLog(prev => {
      const next = { ...prev }
      delete next[dateKey]
      localStorage.setItem(storageKey, JSON.stringify(next))
      return next
    })
  }, [storageKey])

  const onPrevMonth = useCallback(() => {
    setCalView(({ year, month }) =>
      month === 0 ? { year: year - 1, month: 11 } : { year, month: month - 1 }
    )
  }, [])

  const onNextMonth = useCallback(() => {
    setCalView(prev => {
      const now = new Date()
      if (
        prev.year > now.getFullYear() ||
        (prev.year === now.getFullYear() && prev.month >= now.getMonth())
      ) return prev
      return prev.month === 11
        ? { year: prev.year + 1, month: 0 }
        : { year: prev.year, month: prev.month + 1 }
    })
  }, [])

  const onToday = useCallback(() => {
    const now = new Date()
    setCalView({ year: now.getFullYear(), month: now.getMonth() })
  }, [])

  return {
    moodLog,
    streak,
    currentYear:  calView.year,
    currentMonth: calView.month,
    canGoNext,
    onLogMood,
    onRemoveMood,
    onPrevMonth,
    onNextMonth,
    onToday,
  }
}