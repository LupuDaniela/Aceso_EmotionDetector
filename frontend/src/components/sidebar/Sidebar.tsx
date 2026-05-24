import logo from '@/assets/logo_aceso.png'
import type { NavView, AchievementConfig } from '@/types'
import { useAuth } from '@/hooks/useAuth'
import styles from './Sidebar.module.css'

interface Props {
  activeView:       NavView
  onView:           (v: NavView) => void
  streak:           number
  currentCharacter: AchievementConfig | null
}

const NAV_ITEMS: { id: NavView; icon: string; label: string }[] = [
  { id: 'home',         icon: '🏠', label: 'Acasă' },
  { id: 'conversation', icon: '💬', label: 'Conversație nouă' },
  { id: 'history',      icon: '📜', label: 'Conversații avute' },
  { id: 'stats',        icon: '📊', label: 'Statistici' },
  { id: 'calendar',     icon: '📅', label: 'Calendar dispoziție' },
  { id: 'achievements', icon: '🏆', label: 'Realizări & Avataruri' },
  { id: 'settings',     icon: '⚙️', label: 'Setări' },
]

export default function Sidebar({ activeView, onView, streak, currentCharacter }: Props) {
  const { logout } = useAuth()

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <img src={logo} alt="Aceso" className={styles.logoImg} />
      </div>

      <nav className={styles.nav}>
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            type="button"
            className={[
              styles.navItem,
              activeView === item.id ? styles.active : '',
            ].filter(Boolean).join(' ')}
            onClick={() => onView(item.id)}
          >
            <span className={styles.navIcon}>{item.icon}</span>
            <span className={styles.navLabel}>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className={styles.spacer} />

      {currentCharacter && (
        <div className={styles.currentAch}>
          <img src={currentCharacter.img} alt={currentCharacter.name} className={styles.achImg} />
          <div>
            <p className={styles.achName}>{currentCharacter.name}</p>
            <p className={styles.achStreak}>🔥 {streak} zile</p>
          </div>
        </div>
      )}

      <button type="button" className={styles.logoutBtn} onClick={logout}>
        Deconectare
      </button>

      <p className={styles.version}>Aceso EmotionDetector v1.0</p>
    </aside>
  )
}