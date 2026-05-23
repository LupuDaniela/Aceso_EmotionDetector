import { THEMES } from '@/constants/themes'
import type { ThemeKey } from '@/types'
import styles from './ThemePicker.module.css'

interface Props {
  current:  ThemeKey
  onSelect: (key: ThemeKey) => void
}

export default function ThemePicker({ current, onSelect }: Props) {
  return (
    <div className={styles.grid} role="radiogroup" aria-label="Selectează tema">
      {THEMES.map(t => (
        <div key={t.key} className={styles.item}>
          <button
            type="button"
            role="radio"
            aria-checked={current === t.key}
            className={`${styles.btn} ${current === t.key ? styles.selected : ''}`}
            onClick={() => onSelect(t.key)}
            title={t.name}
          >
            <img src={t.bg} alt="" aria-hidden="true" />
          </button>
          <span className={styles.label}>{t.name}</span>
        </div>
      ))}
    </div>
  )
}