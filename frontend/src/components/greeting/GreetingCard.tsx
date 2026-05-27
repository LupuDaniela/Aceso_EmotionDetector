import { useState } from 'react'
import type { AchievementConfig } from '@/types'
import styles from './GreetingCard.module.css'

interface Props {
  userName:          string
  currentCharacter:  AchievementConfig | null
  unlockedAchs:      AchievementConfig[]
  onSelectCharacter: (ach: AchievementConfig) => void
}

export default function GreetingCard({ userName, currentCharacter, unlockedAchs, onSelectCharacter }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false)

  function handleSelect(ach: AchievementConfig) {
    onSelectCharacter(ach)
    setPickerOpen(false)
  }

  return (
    <div className={styles.card}>
      <div className={styles.charWrapper}>
        <button
          type="button"
          className={styles.charSlot}
          onClick={() => unlockedAchs.length > 0 && setPickerOpen(v => !v)}
          aria-label="Alege personaj"
        >
          {currentCharacter
            ? <img src={currentCharacter.img} alt={currentCharacter.name} />
            : <span className={styles.lock}>🔒</span>
          }
        </button>
        {unlockedAchs.length > 0 && (
          <span className={styles.arrow}>▲</span>
        )}
        {pickerOpen && (
          <div className={styles.picker}>
            {unlockedAchs.map(ach => (
              <button
                key={ach.days}
                type="button"
                className={`${styles.pickerBtn} ${currentCharacter?.days === ach.days ? styles.pickerActive : ''}`}
                onClick={() => handleSelect(ach)}
                title={ach.name}
              >
                <img src={ach.img} alt={ach.name} />
                <span>{ach.name}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className={styles.bubble}>
        <p>Bună, <strong className={styles.name}>{userName}</strong>! Cum ți-a fost ziua?</p>
        {currentCharacter
          ? <p className={styles.sub}>{currentCharacter.name}</p>
          : <p className={styles.hint}>Atinge 3 zile streak pentru primul avatar!</p>
        }
      </div>
    </div>
  )
}