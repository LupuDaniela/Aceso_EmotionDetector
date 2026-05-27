import { useMemo } from 'react'
import { ACHIEVEMENTS } from '@/constants/achievements'
import type { AchievementConfig } from '@/types'
import styles from './AchievementsGrid.module.css'

interface Props {
  unlockedAchs:  AchievementConfig[]
  compact?:      boolean
  selectedDays?: number | null
  onSelect?:     (ach: AchievementConfig) => void
}

export default function AchievementsGrid({ unlockedAchs, compact = false, selectedDays, onSelect }: Props) {
  const unlockedDays = useMemo(
    () => new Set(unlockedAchs.map(a => a.days)),
    [unlockedAchs]
  )

  return (
    <div className={styles.grid}>
      {ACHIEVEMENTS.map(ach => {
        const isUnlocked = unlockedDays.has(ach.days)
        const isSelected = isUnlocked && selectedDays === ach.days

        return (
          <div key={ach.days} className={styles.item}>
            <div
              role={isUnlocked && onSelect ? 'button' : undefined}
              tabIndex={isUnlocked && onSelect ? 0 : undefined}
              className={[
                styles.slot,
                compact    ? styles.compact  : '',
                isUnlocked ? styles.unlocked : styles.locked,
                isSelected ? styles.selected : '',
              ].filter(Boolean).join(' ')}
              title={isUnlocked ? (onSelect ? `Selectează ${ach.name}` : ach.name) : `Deblochează la ${ach.days} zile`}
              aria-label={isUnlocked ? ach.name : `Blocat — necesită ${ach.days} zile`}
              aria-pressed={isSelected || undefined}
              onClick={() => isUnlocked && onSelect?.(ach)}
              onKeyDown={e => {
                if ((e.key === 'Enter' || e.key === ' ') && isUnlocked && onSelect) {
                  e.preventDefault()
                  onSelect(ach)
                }
              }}
              style={isUnlocked && onSelect ? { cursor: 'pointer' } : undefined}
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