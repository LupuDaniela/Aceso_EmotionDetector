import { useMemo } from 'react'
import { ACHIEVEMENTS } from '@/constants/achievements'
import type { AchievementConfig } from '@/types'
import styles from './AchievementsGrid.module.css'

interface Props {
  unlockedAchs: AchievementConfig[]
  compact?:     boolean
}

export default function AchievementsGrid({ unlockedAchs, compact = false }: Props) {
  const unlockedDays = useMemo(
    () => new Set(unlockedAchs.map(a => a.days)),
    [unlockedAchs]
  )

  return (
    <div className={styles.grid}>
      {ACHIEVEMENTS.map(ach => {
        const isUnlocked = unlockedDays.has(ach.days)
        return (
          <div key={ach.days} className={styles.item}>
            <div
              className={[
                styles.slot,
                compact    ? styles.compact   : '',
                isUnlocked ? styles.unlocked  : styles.locked,
              ].filter(Boolean).join(' ')}
              title={isUnlocked ? ach.name : `Deblochează la ${ach.days} zile`}
              aria-label={isUnlocked ? ach.name : `Blocat — necesită ${ach.days} zile`}
            >
              {isUnlocked
                ? <img src={ach.img} alt={ach.name} />
                : <span aria-hidden="true">🔒</span>
              }
            </div>
            <span className={styles.label}>{ach.label}</span>
          </div>
        )
      })}
    </div>
  )
}