import logoAceso from '@/assets/logo_aceso.png'
import type { NavView } from '@/types'
import styles from './Sidebar.module.css'

interface NavItem {
  view:  NavView
  icon:  string
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { view: 'home',         icon: '💬', label: 'Conversație nouă'     },
  { view: 'stats',        icon: '📊', label: 'Statistici'            },
  { view: 'calendar',     icon: '📅', label: 'Calendar dispoziție'  },
  { view: 'settings',     icon: '⚙️',  label: 'Setări'               },
  { view: 'achievements', icon: '🏆', label: 'Realizări & Avataruri' },
]

interface Props {
  activeView:  NavView
  sidebarBg:   string
  activeNavBg: string
  onNavigate:  (view: NavView) => void
  onLogout:    () => void
}

export default function Sidebar({
  activeView, sidebarBg, activeNavBg, onNavigate, onLogout,
}: Props) {
  return (
    <nav
      className={styles.sidebar}
      style={{ background: sidebarBg }}
      aria-label="Navigare principală"
    >
      <div className={styles.logoWrap}>
        <img src={logoAceso} alt="Aceso" className={styles.logo} />
      </div>

      <ul className={styles.navList} role="list">
        {NAV_ITEMS.map(({ view, icon, label }) => {
          const isActive = activeView === view
          return (
            <li key={view}>
              <button
                type="button"
                className={`${styles.navItem} ${isActive ? styles.active : ''}`}
                style={isActive ? { background: activeNavBg } : undefined}
                onClick={() => onNavigate(view)}
                aria-current={isActive ? 'page' : undefined}
              >
                <span className={styles.icon} aria-hidden="true">{icon}</span>
                {label}
              </button>
            </li>
          )
        })}
      </ul>

      <footer className={styles.footer}>
        <button
          type="button"
          className={styles.logoutBtn}
          onClick={onLogout}
        >
          Deconectare
        </button>
        <p className={styles.version}>Aceso EmotionDetector v1.0</p>
      </footer>
    </nav>
  )
}