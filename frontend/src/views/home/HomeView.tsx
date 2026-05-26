import MoodCalendar from '@/components/calendar/MoodCalendar'
import AchievementsGrid from '@/components/achievements/AchievementsGrid'
import type { CalendarProps, AchievementConfig } from '@/types'
import styles from './HomeView.module.css'

interface Props extends CalendarProps {
  streak:             number
  unlockedAchs:       AchievementConfig[]
  nextAchievement:    AchievementConfig | null
  progressToNext:     number
  onViewCalendar:     () => void
  onViewAchievements: () => void
}

export default function HomeView({
  streak, unlockedAchs, nextAchievement, progressToNext,
  moodLog, currentYear, currentMonth, canGoNext,
  onLogMood, onRemoveMood, onPrevMonth, onNextMonth, onToday,
  onViewCalendar, onViewAchievements,
}: Props) {
  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <h1 className={styles.title}>Bun venit! 👋</h1>
        <p className={styles.subtitle}>
          {streak === 0
            ? 'Notează prima ta dispoziție de azi pentru a începe streakul.'
            : `${streak} ${streak === 1 ? 'zi' : 'zile'} consecutiv — continuă tot așa! 🎉`}
        </p>
      </div>

      {nextAchievement && (
        <div className={styles.progressCard}>
          <div className={styles.progressHeader}>
            <span className={styles.progressLabel}>
              Spre „{nextAchievement.name}" — {streak}/{nextAchievement.days} zile
            </span>
            <span className={styles.progressPct}>{progressToNext}%</span>
          </div>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progressToNext}%` }} />
          </div>
        </div>
      )}

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>📅 Calendar dispoziție</h2>
          <button type="button" className={styles.linkBtn} onClick={onViewCalendar}>
            Vezi tot →
          </button>
        </div>
        <div className={styles.calendarCard}>
          <MoodCalendar
            moodLog={moodLog}
            currentYear={currentYear}
            currentMonth={currentMonth}
            canGoNext={canGoNext}
            onLogMood={onLogMood}
            onRemoveMood={onRemoveMood}
            onPrevMonth={onPrevMonth}
            onNextMonth={onNextMonth}
            onToday={onToday}
            showLegend={false}
            light
          />
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>🏆 Realizări & Avataruri</h2>
          <button type="button" className={styles.linkBtn} onClick={onViewAchievements}>
            Vezi tot →
          </button>
        </div>

        {unlockedAchs.length === 0 ? (
          <div className={styles.achEmptyCard}>
            <span className={styles.achEmptyIcon}>🔒</span>
            <p className={styles.achEmptyTitle}>Nicio realizare deblocată încă</p>
            <p className={styles.achEmptyHint}>
              Atinge <strong>3 zile</strong> de streak pentru primul avatar!
            </p>
          </div>
        ) : (
          <div className={styles.achCard}>
            <AchievementsGrid unlockedAchs={unlockedAchs} compact />
          </div>
        )}
      </section>
    </div>
  )
}