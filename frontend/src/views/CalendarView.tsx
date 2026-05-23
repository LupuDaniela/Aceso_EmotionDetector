import MoodCalendar        from '@/components/calendar/MoodCalendar'
import Card, { CardTitle } from '@/components/card/Card'
import { MONTHS_RO }       from '@/constants/calendar'
import type { CalendarProps } from '@/types'

interface Props extends CalendarProps {
  currentYear:  number
  currentMonth: number
}

export default function CalendarView(props: Props) {
  return (
    <Card>
      <CardTitle>
        📅 {MONTHS_RO[props.currentMonth]} {props.currentYear}
      </CardTitle>
      <MoodCalendar {...props} showLegend />
    </Card>
  )
}