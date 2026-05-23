import type { AchievementConfig } from '@/types'
import ach1 from '@/assets/achievement1.png'
import ach2 from '@/assets/achievement2.png'
import ach3 from '@/assets/achievement3.png'
import ach4 from '@/assets/achievement4.png'
import ach5 from '@/assets/achievement5.png'
import ach6 from '@/assets/achievement6.png'
import ach7 from '@/assets/achievement7.png'

export const ACHIEVEMENTS: AchievementConfig[] = [
  { days: 3,  img: ach1, label: '3 zile',  name: 'Pisica'     },
  { days: 7,  img: ach2, label: '7 zile',  name: 'Panda'     },
  { days: 14, img: ach3, label: '14 zile', name: 'Catel' },
  { days: 21, img: ach4, label: '21 zile', name: 'Iepure'   },
  { days: 30, img: ach5, label: '1 lună',  name: 'Ponei'   },
  { days: 60, img: ach6, label: '2 luni',  name: 'Lenes'     },
  { days: 90, img: ach7, label: '3 luni',  name: 'Testoasar'       },
]