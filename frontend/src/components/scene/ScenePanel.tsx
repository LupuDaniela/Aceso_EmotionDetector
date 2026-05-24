import type { AcesoTheme } from '@/constants/themes'
import type { AchievementConfig } from '@/types'
import styles from './ScenePanel.module.css'

interface Props {
  theme:            AcesoTheme
  streak:           number
  currentCharacter: AchievementConfig | null
  nextAchievement:  AchievementConfig | null
}

function getGreeting(): string {
  const h = new Date().getHours()
  if (h < 12) return 'Bună dimineața! ☀️'
  if (h < 18) return 'Bună ziua! 🌤️'
  return 'Bună seara! 🌙'
}

export default function ScenePanel({ theme, streak, currentCharacter, nextAchievement }: Props) {
  const daysLeft = nextAchievement ? nextAchievement.days - streak : 0

  return (
    <aside
      className={styles.panel}
      style={{ backgroundImage: `url(${theme.sceneImage})` }}
    >
      <div className={styles.overlay}>
        <div className={styles.characterArea}>
          {currentCharacter ? (
            <img
              src={currentCharacter.img}
              alt={currentCharacter.name}
              className={styles.character}
            />
          ) : (
            <div className={styles.characterPlaceholder}>
              <span className={styles.placeholderIcon}>✨</span>
            </div>
          )}

          <div className={styles.bubble}>
            <p className={styles.bubbleText}>{getGreeting()}</p>
            {!currentCharacter && streak === 0 && (
              <p className={styles.bubbleHint}>
                Notează dispoziția zilnică pentru a debloca primul personaj!
              </p>
            )}
            {!currentCharacter && streak > 0 && nextAchievement && (
              <p className={styles.bubbleHint}>
                Mai {daysLeft} {daysLeft === 1 ? 'zi' : 'zile'} până la primul personaj!
              </p>
            )}
          </div>
        </div>

        <div className={styles.streakBadge}>
          <span>🔥</span>
          <span className={styles.streakNum}>{streak}</span>
          <span className={styles.streakLabel}>zile streak</span>
        </div>
      </div>
    </aside>
  )
}