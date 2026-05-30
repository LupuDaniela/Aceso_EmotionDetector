import { useState, useMemo, useCallback } from 'react'
import { useCalendar }       from './useCalendar'
import { THEMES, DEFAULT_THEME_ID } from '@/constants/themes'
import { ACHIEVEMENTS }      from '@/constants/achievements'
import type { ThemeId, AcesoTheme } from '@/constants/themes'
import type { NavView, AchievementConfig } from '@/types'
import type { UseCalendarReturn } from './useCalendar'

export interface DashboardState {
  theme:               AcesoTheme
  activeView:          NavView
  streak:              number
  unlockedAchs:        AchievementConfig[]
  currentCharacter:    AchievementConfig | null
  nextAchievement:     AchievementConfig | null
  progressToNext:      number
  keepAchievements:    boolean
  setTheme:            (id: ThemeId) => void
  setActiveView:       (view: NavView) => void
  setSelectedCharacter:(ach: AchievementConfig) => void
  setKeepAchievements: (val: boolean) => void
  calendar:            UseCalendarReturn
}

export function useDashboard(userId?: string | number): DashboardState {
  const storageKey   = userId ? `aceso_moods_${userId}` : 'aceso_mood_log'
  const charKey      = userId ? `aceso_character_${userId}` : 'aceso_character'
  const themeKey     = userId ? `aceso_theme_${userId}` : 'aceso_theme'
  const keepAchKey   = userId ? `aceso_keep_ach_${userId}` : 'aceso_keep_ach'
  const maxStreakKey  = userId ? `aceso_max_streak_${userId}` : 'aceso_max_streak'
  const calendar     = useCalendar(storageKey)

  const [themeId, setThemeId] = useState<ThemeId>(() => {
    const saved = localStorage.getItem(themeKey)
    return (saved as ThemeId) ?? DEFAULT_THEME_ID
  })

  const [activeView, setActiveView] = useState<NavView>('home')

  const [selectedDays, setSelectedDays] = useState<number | null>(() => {
    const saved = localStorage.getItem(charKey)
    return saved ? Number(saved) : null
  })

  const [keepAchievements, setKeepAchievementsState] = useState<boolean>(() => {
    const saved = localStorage.getItem(keepAchKey)
    return saved === null ? true : saved === 'true'
  })

  const theme  = THEMES[themeId]
  const streak = calendar.streak

  const maxStreak = useMemo(() => {
    const saved = parseInt(localStorage.getItem(maxStreakKey) ?? '0', 10)
    const current = Math.max(saved, streak)
    if (current > saved) localStorage.setItem(maxStreakKey, String(current))
    return current
  }, [streak, maxStreakKey])

  const unlockedAchs = useMemo(
    () => ACHIEVEMENTS.filter(a => (keepAchievements ? maxStreak : streak) >= a.days),
    [streak, maxStreak, keepAchievements]
  )

  const currentCharacter = useMemo(() => {
    if (selectedDays !== null) {
      const found = unlockedAchs.find(a => a.days === selectedDays)
      if (found) return found
    }
    return unlockedAchs.at(-1) ?? null
  }, [unlockedAchs, selectedDays])

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

  const setTheme = useCallback((id: ThemeId) => {
    setThemeId(id)
    localStorage.setItem(themeKey, id)
  }, [themeKey])

  const setSelectedCharacter = useCallback((ach: AchievementConfig) => {
    setSelectedDays(ach.days)
    localStorage.setItem(charKey, String(ach.days))
  }, [charKey])

  const setKeepAchievements = useCallback((val: boolean) => {
    setKeepAchievementsState(val)
    localStorage.setItem(keepAchKey, String(val))
  }, [keepAchKey])

  return {
    theme, activeView, streak,
    unlockedAchs, currentCharacter, nextAchievement, progressToNext,
    keepAchievements,
    setTheme, setActiveView, setSelectedCharacter, setKeepAchievements,
    calendar,
  }
}