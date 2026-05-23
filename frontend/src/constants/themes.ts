import type { ThemeConfig, ThemeKey } from '@/types'
import bgPurple from '@/assets/option1.png'
import bgSunset from '@/assets/option2.png'
import bgForest from '@/assets/option3.png'
import bgGalaxy from '@/assets/option4.png'

export const THEMES: ThemeConfig[] = [
  {
    key:         'purple',
    name:        'Noapte violetă',
    bg:          bgPurple,
    bgColor:     '#2b1a50',
    sidebarBg:   'rgba(25,10,50,0.88)',
    activeNavBg: 'rgba(123,104,238,0.40)',
    fillColor:   '#9b7fe8',
  },
  {
    key:         'sunset',
    name:        'Apus plajă',
    bg:          bgSunset,
    bgColor:     '#4a1e05',
    sidebarBg:   'rgba(75,28,5,0.88)',
    activeNavBg: 'rgba(204,96,34,0.40)',
    fillColor:   '#e87c40',
  },
  {
    key:         'forest',
    name:        'Pădure nocturnă',
    bg:          bgForest,
    bgColor:     '#0d2b18',
    sidebarBg:   'rgba(12,40,22,0.90)',
    activeNavBg: 'rgba(45,106,76,0.40)',
    fillColor:   '#52b788',
  },
  {
    key:         'galaxy',
    name:        'Galaxie',
    bg:          bgGalaxy,
    bgColor:     '#080828',
    sidebarBg:   'rgba(8,8,40,0.92)',
    activeNavBg: 'rgba(61,95,160,0.40)',
    fillColor:   '#6b8dd6',
  },
]

export function getTheme(key: ThemeKey | string): ThemeConfig {
  return THEMES.find(t => t.key === key) ?? THEMES[0]
}