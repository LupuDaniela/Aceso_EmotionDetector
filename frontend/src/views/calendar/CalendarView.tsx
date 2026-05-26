import MoodCalendar from '@/components/calendar/MoodCalendar'
import type { CalendarProps } from '@/types'
import styles from './CalendarView.module.css'

export default function CalendarView(props: CalendarProps) {
  return (
    <div className={styles.root}>
      <h1 className={styles.title}>📅 Calendar dispoziție</h1>
      <div className={styles.card}>
        <MoodCalendar {...props} showLegend light />
      </div>
    </div>
  )
}