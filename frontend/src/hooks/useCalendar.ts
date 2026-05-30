import { useState, useMemo, useCallback, useEffect } from 'react'
import { toDateKey } from '@/utils/date'
import type { MoodKey, CalendarProps } from '@/types'
import { API_URL } from '../utils/api'

type MoodLog = Record<string, MoodKey>

function calculateStreak(moodLog: MoodLog): number {
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const todayKey     = toDateKey(today)
  const yesterdayKey = toDateKey(new Date(today.getTime() - 86400000))

  const startKey = moodLog[todayKey]
    ? todayKey
    : moodLog[yesterdayKey]
    ? yesterdayKey
    : null

  if (!startKey) return 0

  const cursor = new Date(startKey)
  let streak   = 0

  while (moodLog[toDateKey(cursor)]) {
    streak++
    cursor.setDate(cursor.getDate() - 1)
  }

  return streak
}

export interface UseCalendarReturn extends CalendarProps {
  streak: number
}

export function useCalendar(userId: string | number | null | undefined): UseCalendarReturn {
  const [moodLog, setMoodLog] = useState<MoodLog>({})
  const [calView, setCalView] = useState(() => {
    const now = new Date()
    return { year: now.getFullYear(), month: now.getMonth() }
  })

  useEffect(() => {
    if (!userId) {
      setMoodLog({})
      return
    }

    const token = localStorage.getItem('aceso_token')
    if (!token) return

    setMoodLog({})

    fetch(`${API_URL}/api/moods`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : {})
      .then((data: MoodLog) => setMoodLog(data))
      .catch(() => {})
  }, [userId])

  const streak = useMemo(() => calculateStreak(moodLog), [moodLog])

  const canGoNext = useMemo(() => {
    const now = new Date()
    return (
      calView.year < now.getFullYear() ||
      (calView.year === now.getFullYear() && calView.month < now.getMonth())
    )
  }, [calView])

  const onLogMood = useCallback((dateKey: string, mood: MoodKey) => {
    setMoodLog(prev => ({ ...prev, [dateKey]: mood }))
    const token = localStorage.getItem('aceso_token')
    if (!token) return
    fetch('${API_URL}/api/moods', {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ date_key: dateKey, mood_key: mood }),
    }).catch(() => {})
  }, [])

  const onRemoveMood = useCallback((dateKey: string) => {
    setMoodLog(prev => {
      const next = { ...prev }
      delete next[dateKey]
      return next
    })
    const token = localStorage.getItem('aceso_token')
    if (!token) return
    fetch(`${API_URL}/api/moods/${dateKey}`, {
      method:  'DELETE',
      headers: { 'Authorization': `Bearer ${token}` },
    }).catch(() => {})
  }, [])

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