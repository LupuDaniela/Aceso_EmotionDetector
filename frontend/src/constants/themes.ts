import option1 from '@/assets/option1.png'
import option2 from '@/assets/option2.png'
import option3 from '@/assets/option3.png'
import option4 from '@/assets/option4.png'

export type ThemeId = 'purple' | 'sunset' | 'forest' | 'galaxy'

export interface AcesoTheme {
  id:          ThemeId
  name:        string
  label:       string
  accent:      string
  accentDark:  string
  accentLight: string
  sceneImage:  string
}

export const THEMES: Record<ThemeId, AcesoTheme> = {
  purple: {
    id:          'purple',
    name:        'Purple Night',
    label:       '🌙 Purple',
    accent:      '#8B6EC5',
    accentDark:  '#7050A8',
    accentLight: '#EDE8F8',
    sceneImage:  option1,
  },
  sunset: {
    id:          'sunset',
    name:        'Sunset',
    label:       '🌅 Sunset',
    accent:      '#D4763A',
    accentDark:  '#B85E28',
    accentLight: '#FAEEE5',
    sceneImage:  option2,
  },
  forest: {
    id:          'forest',
    name:        'Forest',
    label:       '🌿 Forest',
    accent:      '#3A8A5A',
    accentDark:  '#2A6E44',
    accentLight: '#E4F2EA',
    sceneImage:  option3,
  },
  galaxy: {
    id:          'galaxy',
    name:        'Galaxy',
    label:       '🌌 Galaxy',
    accent:      '#5C6BC0',
    accentDark:  '#3F51B5',
    accentLight: '#E8EAFF',
    sceneImage:  option4,
  },
}

export const DEFAULT_THEME_ID: ThemeId = 'purple'