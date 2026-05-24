import { useState, useMemo, useCallback } from 'react'
import { useCalendar } from './useCalendar'
import { THEMES, DEFAULT_THEME_ID, type ThemeId, type AcesoTheme } from '@/constants/themes'
import { ACHIEVEMENTS } from '@/constants/achievements'
import type { NavView, AchievementConfig } from '@/types'
import type { UseCalendarReturn } from './useCalendar'

export interface DashboardState {
  theme:            AcesoTheme
  activeView:       NavView
  streak:           number
  unlockedAchs:     AchievementConfig[]
  currentCharacter: AchievementConfig | null
  nextAchievement:  AchievementConfig | null
  progressToNext:   number
  setTheme:         (id: ThemeId) => void
  setActiveView:    (view: NavView) => void
  calendar:         UseCalendarReturn
}

export function useDashboard(userId?: string | number): DashboardState {
  const calendar = useCalendar(userId ?? null)

  const [themeId, setThemeId] = useState<ThemeId>(() => {
    const saved = localStorage.getItem('aceso_theme') as ThemeId | null
    return saved && saved in THEMES ? saved : DEFAULT_THEME_ID
  })
  const [activeView, setActiveView] = useState<NavView>('home')

  const theme  = useMemo(() => THEMES[themeId] ?? THEMES[DEFAULT_THEME_ID], [themeId])
  const streak = calendar.streak

  const unlockedAchs = useMemo(
    () => ACHIEVEMENTS.filter(a => streak >= a.days),
    [streak]
  )

  const currentCharacter = useMemo(() => unlockedAchs.at(-1) ?? null, [unlockedAchs])

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
    localStorage.setItem('aceso_theme', id)
  }, [])

  return {
    theme, activeView, streak,
    unlockedAchs, currentCharacter, nextAchievement, progressToNext,
    setTheme, setActiveView, calendar,
  }
}