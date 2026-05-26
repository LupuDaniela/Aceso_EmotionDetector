import { ACHIEVEMENTS } from '@/constants/achievements'
import type { AchievementConfig } from '@/types'
import styles from './Achievements.module.css'

interface Props {
  unlockedAchs: AchievementConfig[]
  streak:       number
}

export default function AchievementsView({ unlockedAchs, streak }: Props) {
  const unlockedDays = new Set(unlockedAchs.map(a => a.days))

  return (
    <div className={styles.root}>
      <h1 className={styles.title}>🏆 Realizări & Avataruri</h1>

      <div className={styles.grid}>
        {ACHIEVEMENTS.map(ach => {
          const isUnlocked = unlockedDays.has(ach.days)
          return (
            <div key={ach.days} className={styles.item}>
              <div
                className={[styles.slot, isUnlocked ? styles.unlocked : styles.locked].join(' ')}
                title={isUnlocked ? ach.name : `Deblochează la ${ach.days} zile`}
              >
                {isUnlocked
                  ? <img src={ach.img} alt={ach.name} />
                  : <span>🔒</span>
                }
              </div>
              <span className={styles.label}>{ach.label}</span>
            </div>
          )
        })}
      </div>

      <div className={styles.infoCard}>
        <p className={styles.infoTitle}>ℹ️ Cum deblochezi avataruri</p>
        {ACHIEVEMENTS.map(ach => (
          <div key={ach.days} className={styles.infoItem}>
            <span className={styles.infoIcon}>
              {unlockedDays.has(ach.days) ? '✅' : '🔒'}
            </span>
            <span className={styles.infoText}>
              {ach.days} zile — <span className={styles.infoName}>{ach.name}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}