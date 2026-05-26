import React from 'react'
import { THEMES, type ThemeId } from '@/constants/themes'
import styles from './SettingsView.module.css'

interface Props {
  themeId:       ThemeId
  onThemeChange: (id: ThemeId) => void
}

export default function SettingsView({ themeId, onThemeChange }: Props) {
  return (
    <div>
      <div className={styles.themeSection}>
        <h3 className={styles.sectionTitle}>🎨 Temă vizuală</h3>
        <p className={styles.sectionHint}>
          Schimbă tema pentru a personaliza culorile și fundalul scenic.
        </p>
        <div className={styles.themePicker}>
          {Object.values(THEMES).map(theme => (
            <button
              key={theme.id}
              type="button"
              className={[
                styles.themeBtn,
                theme.id === themeId ? styles.themeActive : '',
              ].filter(Boolean).join(' ')}
              style={{ '--btn-accent': theme.accent } as React.CSSProperties}
              onClick={() => onThemeChange(theme.id)}
              title={theme.name}
            >
              <span
                className={styles.themeBtnPreview}
                style={{ backgroundImage: `url(${theme.sceneImage})` }}
              />
              <span className={styles.themeBtnLabel}>{theme.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}