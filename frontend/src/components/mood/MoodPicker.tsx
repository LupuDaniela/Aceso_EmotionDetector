import { createPortal } from 'react-dom'
import { MOOD_CONFIG, MONTHS_RO } from '@/constants/calendar'
import type { MoodKey } from '@/types'
import styles from './MoodPicker.module.css'

interface Props {
  dateKey:     string
  currentMood: MoodKey | undefined
  onSelect:    (mood: MoodKey) => void
  onRemove:    () => void
  onClose:     () => void
}

function formatLabel(dateKey: string): string {
  const [y, m, d] = dateKey.split('-').map(Number)
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const isToday = new Date(y, m - 1, d).getTime() === today.getTime()
  return isToday
    ? `astăzi, ${d} ${MONTHS_RO[m - 1]}`
    : `${d} ${MONTHS_RO[m - 1]} ${y}`
}

export default function MoodPicker({
  dateKey, currentMood, onSelect, onRemove, onClose,
}: Props) {
  return createPortal(
    <div
      className={styles.backdrop}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <h3 className={styles.title}>
            Cum te-ai simțit{' '}
            <span className={styles.dateLabel}>{formatLabel(dateKey)}</span>?
          </h3>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Închide"
          >
            ✕
          </button>
        </div>

        <div className={styles.grid}>
          {(Object.entries(MOOD_CONFIG) as [MoodKey, typeof MOOD_CONFIG[MoodKey]][]).map(
            ([key, { emoji, label, color }]) => (
              <button
                key={key}
                type="button"
                className={`${styles.moodBtn} ${currentMood === key ? styles.selected : ''}`}
                style={currentMood === key ? { background: color, borderColor: 'transparent' } : undefined}
                onClick={() => onSelect(key)}
                title={label}
              >
                <span className={styles.emoji}>{emoji}</span>
                <span className={styles.moodLabel}>{label}</span>
              </button>
            )
          )}
        </div>

        {currentMood && (
          <button type="button" className={styles.removeBtn} onClick={onRemove}>
            Șterge dispoziția
          </button>
        )}
      </div>
    </div>,
    document.body
  )
}