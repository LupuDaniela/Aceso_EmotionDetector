import { useAuthContext } from '@/context/AuthContext'
import { useDashboard }   from '@/hooks/useDashboard'
import Sidebar            from '@/components/sidebar/Sidebar'
import HomeView           from '@/views/HomeView'
import StatsView          from '@/views/StatsView'
import CalendarView       from '@/views/CalendarView'
import SettingsView       from '@/views/SettingsView'
import AchievementsView   from '@/views/AchievementsView'
import type { NavView }   from '@/types'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { user, loading, logout } = useAuthContext()
  const db = useDashboard(user?.id)

  if (loading) {
    return (
      <div className={styles.loader}>
        <span className={styles.loaderSpinner} />
      </div>
    )
  }

  const userName = user?.name ?? 'Utilizator'

  function renderView(view: NavView) {
    switch (view) {
      case 'home':
        return (
          <HomeView
            userName={userName}
            streak={db.streak}
            currentCharacter={db.currentCharacter}
            nextAchievement={db.nextAchievement}
            progressToNext={db.progressToNext}
            fillColor={db.theme.fillColor}
            unlockedAchs={db.unlockedAchs}
            calendar={db.calendar}
          />
        )
      case 'stats':
        return (
          <StatsView
            streak={db.streak}
            unlockedCount={db.unlockedAchs.length}
            calendar={db.calendar}
          />
        )
      case 'calendar':
        return <CalendarView {...db.calendar} />
      case 'settings':
        return (
          <SettingsView
            currentTheme={db.theme.key}
            userName={userName}
            streak={db.streak}
            onSetTheme={db.setTheme}
          />
        )
      case 'achievements':
        return (
          <AchievementsView
            streak={db.streak}
            unlockedAchs={db.unlockedAchs}
          />
        )
    }
  }

  return (
    <div className={styles.root} style={{ backgroundColor: db.theme.bgColor }}>
      <div
        className={styles.bg}
        style={{ backgroundImage: `url(${db.theme.bg})` }}
        aria-hidden="true"
      />
      <div className={styles.overlay} aria-hidden="true" />

      <Sidebar
        activeView={db.activeView}
        sidebarBg={db.theme.sidebarBg}
        activeNavBg={db.theme.activeNavBg}
        onNavigate={db.setActiveView}
        onLogout={logout}
      />

      <main className={styles.main}>
        {renderView(db.activeView)}
      </main>
    </div>
  )
}