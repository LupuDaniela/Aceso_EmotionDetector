import ProgressBar from '@/components/progress/ProgressBar'
import type { AchievementConfig } from '@/types'
import styles from './StreakBar.module.css'

interface Props {
  streak:          number
  nextAchievement: AchievementConfig | null
  progressToNext:  number
  fillColor:       string
}

export default function StreakBar({
  streak, nextAchievement, progressToNext, fillColor,
}: Props) {
  return (
    <div className={styles.wrap}>
      <div className={styles.row}>
        <div className={styles.info}>
          <span aria-hidden="true">🔥</span>
          <span className={styles.count}>{streak}</span>
          <span className={styles.sub}>
            {nextAchievement ? 'zile streak' : 'zile · toate deblocate 🎉'}
          </span>
        </div>
        <p className={styles.hint}>
          Notează o dispoziție zilnic pentru a crește streak-ul
        </p>
      </div>

      {nextAchievement && (
        <div className={styles.progress}>
          <ProgressBar
            value={progressToNext}
            color={fillColor}
            label={`Spre „${nextAchievement.name}" — ${streak}/${nextAchievement.days} zile (${progressToNext}%)`}
          />
        </div>
      )}
    </div>
  )
}