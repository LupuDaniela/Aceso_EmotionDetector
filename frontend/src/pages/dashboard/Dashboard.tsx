import { useEffect } from 'react'
import { useDashboard } from '@/hooks/useDashboard'
import { useAuth } from '@/hooks/useAuth'
import Sidebar from '@/components/sidebar/Sidebar'
import HomeView from '../../views/HomeView'
import StatsView from '../../views/StatsView'
import CalendarView from '../../views/CalendarView'
import AchievementsView from '../../views/AchievementsView'
import SettingsView from '../../views/SettingsView'
import ConversationView from '../../views/ConversationView'
import HistoryView from '../../views/HistoryView'
import ScenePanel from '@/components/scene/ScenePanel'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { user, loading } = useAuth()
  const dash = useDashboard(user?.id)
  const {
    theme, activeView, setActiveView,
    streak, unlockedAchs, currentCharacter, nextAchievement, progressToNext,
    setTheme, calendar,
  } = dash

  useEffect(() => {
    if (!theme) return
    const r = document.documentElement
    r.style.setProperty('--accent',       theme.accent)
    r.style.setProperty('--accent-dark',  theme.accentDark)
    r.style.setProperty('--accent-light', theme.accentLight)
  }, [theme])

  if (loading || !theme) return null

  const calProps = {
    moodLog:      calendar.moodLog,
    currentYear:  calendar.currentYear,
    currentMonth: calendar.currentMonth,
    canGoNext:    calendar.canGoNext,
    onLogMood:    calendar.onLogMood,
    onRemoveMood: calendar.onRemoveMood,
    onPrevMonth:  calendar.onPrevMonth,
    onNextMonth:  calendar.onNextMonth,
    onToday:      calendar.onToday,
  }

  const showScene = activeView !== 'conversation' && activeView !== 'history'

  return (
    <div className={styles.layout}>
      <Sidebar
        activeView={activeView}
        onView={setActiveView}
        streak={streak}
        currentCharacter={currentCharacter}
      />

      <main className={styles.main}>
        <div className={styles.content}>
          {activeView === 'home' && (
            <HomeView
              streak={streak}
              unlockedAchs={unlockedAchs}
              nextAchievement={nextAchievement}
              progressToNext={progressToNext}
              onViewCalendar={() => setActiveView('calendar')}
              onViewAchievements={() => setActiveView('achievements')}
              {...calProps}
            />
          )}
          {activeView === 'conversation' && (
            <ConversationView
              theme={theme}
              onLogMood={calendar.onLogMood}
            />
          )}
          {activeView === 'history' && (
            <HistoryView theme={theme} />
          )}
          {activeView === 'stats' && (
            <StatsView
              streak={streak}
              unlockedCount={unlockedAchs.length}
              calendar={calProps}
            />
          )}
          {activeView === 'calendar' && (
            <CalendarView {...calProps} />
          )}
          {activeView === 'achievements' && (
            <AchievementsView unlockedAchs={unlockedAchs} streak={streak} />
          )}
          {activeView === 'settings' && (
            <SettingsView themeId={theme.id} onThemeChange={setTheme} />
          )}
        </div>

        {showScene && (
          <ScenePanel
            theme={theme}
            streak={streak}
            currentCharacter={currentCharacter}
            nextAchievement={nextAchievement}
          />
        )}
      </main>
    </div>
  )
}