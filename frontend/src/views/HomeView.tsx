import GreetingCard        from '@/components/greeting/GreetingCard'
import StreakBar            from '@/components/streak/StreakBar'
import MoodCalendar        from '@/components/calendar/MoodCalendar'
import AchievementsGrid    from '@/components/achievements/AchievementsGrid'
import Card, { CardTitle } from '@/components/card/Card'
import { MONTHS_RO }       from '@/constants/calendar'
import type { AchievementConfig, CalendarProps } from '@/types'

interface Props {
  userName:         string
  streak:           number
  currentCharacter: AchievementConfig | null
  nextAchievement:  AchievementConfig | null
  progressToNext:   number
  fillColor:        string
  unlockedAchs:     AchievementConfig[]
  calendar:         CalendarProps & { currentYear: number; currentMonth: number }
}

export default function HomeView({
  userName, streak, currentCharacter, nextAchievement,
  progressToNext, fillColor, unlockedAchs, calendar,
}: Props) {
  return (
    <>
      <GreetingCard userName={userName} currentCharacter={currentCharacter} />

      <StreakBar
        streak={streak}
        nextAchievement={nextAchievement}
        progressToNext={progressToNext}
        fillColor={fillColor}
      />

      <Card>
        <CardTitle>
          📅 {MONTHS_RO[calendar.currentMonth]} {calendar.currentYear}
        </CardTitle>
        <MoodCalendar {...calendar} showLegend />
      </Card>

      <Card>
        <CardTitle>🏆 Realizări deblocate</CardTitle>
        <AchievementsGrid unlockedAchs={unlockedAchs} compact />
      </Card>
    </>
  )
}