import AchievementsGrid    from '@/components/achievements/AchievementsGrid'
import Card, { CardTitle } from '@/components/card/Card'
import { ACHIEVEMENTS }    from '@/constants/achievements'
import type { AchievementConfig } from '@/types'
import styles from '../views/Achievements.module.css'

interface Props {
  streak:       number
  unlockedAchs: AchievementConfig[]
}

export default function AchievementsView({ streak, unlockedAchs }: Props) {
  return (
    <>
      <Card>
        <CardTitle>🏆 Realizări &amp; Avataruri</CardTitle>
        <AchievementsGrid unlockedAchs={unlockedAchs} />
      </Card>

      <Card>
        <CardTitle>ℹ️ Cum deblochezi avataruri</CardTitle>
        <div className={styles.list}>
          {ACHIEVEMENTS.map(a => {
            const done = streak >= a.days
            return (
              <div
                key={a.days}
                className={`${styles.row} ${done ? styles.done : ''}`}
              >
                <span className={styles.check}>{done ? '✅' : '🔒'}</span>
                <div>
                  <span className={styles.label}>{a.label}</span>
                  {' — '}
                  <span className={styles.name}>{a.name}</span>
                </div>
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )
}