import type { AchievementConfig } from '@/types'
import styles from './GreetingCard.module.css'

interface Props {
  userName:         string
  currentCharacter: AchievementConfig | null
}

export default function GreetingCard({ userName, currentCharacter }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.charSlot}>
        {currentCharacter
          ? <img src={currentCharacter.img} alt={currentCharacter.name} />
          : <span className={styles.lock} aria-label="Blocat">🔒</span>
        }
      </div>

      <div className={styles.bubble}>
        <p>
          Bună, <strong className={styles.name}>{userName}</strong>! Cum ți-a fost ziua? 
        </p>
        {currentCharacter
          ? <p className={styles.sub}>{currentCharacter.name}</p>
          : <p className={styles.hint}>Atinge 3 zile streak pentru primul avatar!</p>
        }
      </div>
    </div>
  )
}