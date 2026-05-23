import { useState, useMemo, useCallback } from 'react'
import { useCalendar }  from './useCalendar'
import { getTheme }     from '@/constants/themes'
import { ACHIEVEMENTS } from '@/constants/achievements'
import type { ThemeKey, NavView, ThemeConfig, AchievementConfig } from '@/types'
import type { UseCalendarReturn } from './useCalendar'

export interface DashboardState {
  theme:            ThemeConfig
  activeView:       NavView
  streak:           number
  unlockedAchs:     AchievementConfig[]
  currentCharacter: AchievementConfig | null
  nextAchievement:  AchievementConfig | null
  progressToNext:   number
  setTheme:         (key: ThemeKey) => void
  setActiveView:    (view: NavView) => void
  calendar:         UseCalendarReturn
}

export function useDashboard(userId?: string | number): DashboardState {
  const storageKey = userId ? `aceso_moods_${userId}` : 'aceso_mood_log'
  const calendar   = useCalendar(storageKey)

  const [themeKey,   setThemeKey]   = useState<ThemeKey>('purple')
  const [activeView, setActiveView] = useState<NavView>('home')

  const theme  = useMemo(() => getTheme(themeKey), [themeKey])
  const streak = calendar.streak

  const unlockedAchs = useMemo(
    () => ACHIEVEMENTS.filter(a => streak >= a.days),
    [streak]
  )

  const currentCharacter = useMemo(
    () => unlockedAchs.at(-1) ?? null,
    [unlockedAchs]
  )

  const nextAchievement = useMemo(
    () => ACHIEVEMENTS.find(a => streak < a.days) ?? null,
    [streak]
  )

  const progressToNext = useMemo(() => {
    if (!nextAchievement) return 100
    const from  = unlockedAchs.at(-1)?.days ?? 0
    const range = nextAchievement.days - from
    return range > 0 ? Math.round(((streak - from) / range) * 100) : 0
  }, [streak, nextAchievement, unlockedAchs])

  const setTheme = useCallback((key: ThemeKey) => setThemeKey(key), [])

  return {
    theme, activeView, streak,
    unlockedAchs, currentCharacter, nextAchievement, progressToNext,
    setTheme, setActiveView, calendar,
  }
}