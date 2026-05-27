import { ACHIEVEMENTS }    from '@/constants/achievements'
import type { AchievementConfig } from '@/types'
import styles from './Achievements.module.css'

interface Props {
  streak:       number
  unlockedAchs: AchievementConfig[]
}

export default function AchievementsView({ streak, unlockedAchs }: Props) {
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
                className={`${styles.slot} ${isUnlocked ? styles.unlocked : styles.locked}`}
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
        <p className={styles.infoTitle}>Cum deblochezi avataruri</p>
        {ACHIEVEMENTS.map(a => (
          <div key={a.days} className={styles.infoItem}>
            <span className={styles.infoIcon}>{streak >= a.days ? '✅' : '🔒'}</span>
            <span className={styles.infoText}>
              {a.label} — <span className={styles.infoName}>{a.name}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}