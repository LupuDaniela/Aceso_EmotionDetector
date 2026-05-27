import { useAuth } from '@/hooks/useAuth'
import logoAceso from '@/assets/logo_aceso.png'
import type { NavView, AchievementConfig } from '@/types'
import styles from './Sidebar.module.css'

interface NavItem { view: NavView; icon: string; label: string }

const NAV_ITEMS: NavItem[] = [
  { view: 'home',         icon: '🏠', label: 'Acasă'            },
  { view: 'conversation', icon: '💬', label: 'Conversație nouă' },
  { view: 'history',      icon: '📜', label: 'Istoric'           },
  { view: 'stats',        icon: '📊', label: 'Statistici'        },
  { view: 'calendar',     icon: '📅', label: 'Calendar'          },
  { view: 'achievements', icon: '🏆', label: 'Realizări'         },
  { view: 'settings',     icon: '⚙️',  label: 'Setări'            },
]

interface Props {
  activeView:       NavView
  onView:           (view: NavView) => void
  streak:           number
  currentCharacter: AchievementConfig | null
}

export default function Sidebar({ activeView, onView, streak, currentCharacter }: Props) {
  const { logout } = useAuth()

  return (
    <nav className={styles.sidebar}>
      <div className={styles.logo}>
        <img src={logoAceso} alt="Aceso" className={styles.logoImg} />
      </div>

      <div className={styles.nav}>
        {NAV_ITEMS.map(({ view, icon, label }) => (
          <button
            key={view}
            type="button"
            className={`${styles.navItem} ${activeView === view ? styles.active : ''}`}
            onClick={() => onView(view)}
          >
            <span className={styles.navIcon}>{icon}</span>
            <span className={styles.navLabel}>{label}</span>
          </button>
        ))}
      </div>

      <div className={styles.spacer} />

      {currentCharacter && (
        <div className={styles.currentAch}>
          <img src={currentCharacter.img} alt={currentCharacter.name} className={styles.achImg} />
          <div>
            <div className={styles.achName}>{currentCharacter.name}</div>
            <div className={styles.achStreak}>🔥 {streak} zile</div>
          </div>
        </div>
      )}

      <button type="button" className={styles.logoutBtn} onClick={logout}>
        Deconectare
      </button>
      <p className={styles.version}>Aceso EmotionDetector v1.0</p>
    </nav>
  )
}