import { THEMES }           from '@/constants/themes'
import type { ThemeId, AcesoTheme } from '@/constants/themes'
import styles from './ThemePicker.module.css'

interface Props {
  current:  ThemeId
  onSelect: (id: ThemeId) => void
}

export default function ThemePicker({ current, onSelect }: Props) {
  const themes = Object.values(THEMES) as AcesoTheme[]

  return (
    <div className={styles.grid} role="radiogroup" aria-label="Selectează tema">
      {themes.map(t => (
        <div key={t.id} className={styles.item}>
          <button
            type="button"
            role="radio"
            aria-checked={current === t.id}
            className={`${styles.btn} ${current === t.id ? styles.selected : ''}`}
            onClick={() => onSelect(t.id)}
            title={t.label}
          >
            <img src={t.sceneImage} alt="" aria-hidden="true" />
          </button>
          <span className={styles.label}>{t.label}</span>
        </div>
      ))}
    </div>
  )
}