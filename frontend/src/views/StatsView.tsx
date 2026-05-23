import MoodCalendar        from '@/components/calendar/MoodCalendar'
import Card, { CardTitle } from '@/components/card/Card'
import { MONTHS_RO }       from '@/constants/calendar'
import { ACHIEVEMENTS }    from '@/constants/achievements'
import type { CalendarProps } from '@/types'
import styles from './StatsView.module.css'

interface Props {
  streak:        number
  unlockedCount: number
  calendar:      CalendarProps & { currentYear: number; currentMonth: number }
}

export default function StatsView({ streak, unlockedCount, calendar }: Props) {
  const stats = [
    { val: streak,                                    lbl: 'Streak curent (zile)'  },
    { val: `${unlockedCount}/${ACHIEVEMENTS.length}`, lbl: 'Avataruri deblocate'   },
    { val: Object.keys(calendar.moodLog).length,      lbl: 'Zile notate total'     },
    { val: '—',                                       lbl: 'Emoție dominantă'      },
  ]

  return (
    <>
      <Card>
        <CardTitle>📊 Statistici generale</CardTitle>
        <div className={styles.grid}>
          {stats.map(s => (
            <div key={s.lbl} className={styles.statCard}>
              <div className={styles.val}>{s.val}</div>
              <div className={styles.lbl}>{s.lbl}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <CardTitle>
          📅 {MONTHS_RO[calendar.currentMonth]} {calendar.currentYear}
        </CardTitle>
        <MoodCalendar {...calendar} showLegend />
      </Card>
    </>
  )
}