import ThemePicker         from '@/components/theme/ThemePicker'
import Card, { CardTitle } from '@/components/card/Card'
import type { ThemeKey }   from '@/types'
import styles from './SettingsView.module.css'

interface Props {
  currentTheme: ThemeKey
  userName:     string
  streak:       number
  onSetTheme:   (key: ThemeKey) => void
}

export default function SettingsView({
  currentTheme, userName, streak, onSetTheme,
}: Props) {
  const rows = [
    { key: 'Utilizator', val: userName         },
    { key: 'Streak',     val: `${streak} zile` },
    { key: 'Limbă',      val: 'Română'         },
  ]

  return (
    <>
      <Card>
        <CardTitle>🎨 Temă / Fundal</CardTitle>
        <ThemePicker current={currentTheme} onSelect={onSetTheme} />
      </Card>

      <Card>
        <CardTitle>👤 Profil</CardTitle>
        <div className={styles.list}>
          {rows.map(r => (
            <div key={r.key} className={styles.row}>
              <span className={styles.key}>{r.key}</span>
              <span className={styles.val}>{r.val}</span>
            </div>
          ))}
        </div>
      </Card>
    </>
  )
}