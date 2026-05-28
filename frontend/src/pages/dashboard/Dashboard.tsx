import { useEffect, useState } from 'react'
import { useDashboard } from '@/hooks/useDashboard'
import { useAuth } from '@/hooks/useAuth'
import Sidebar from '@/components/sidebar/Sidebar'
import HomeView from '../../views/home/HomeView'
import StatsView from '../../views/statistics/StatsView'
import CalendarView from '../../views/calendar/CalendarView'
import AchievementsView from '../../views/achievements/AchievementsView'
import SettingsView from '../../views/settings/SettingsView'
import ConversationView from '../../views/conversation/ConversationView'
import HistoryView from '../../views/history/HistoryView'
import ThreadDetailView from '../../views/history/ThreadDetailView'
import ScenePanel from '@/components/scene/ScenePanel'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { user, loading } = useAuth()
  const dash = useDashboard(user?.id)
  const {
    theme, activeView, setActiveView,
    streak, unlockedAchs, currentCharacter, nextAchievement, progressToNext,
    setTheme, calendar, setSelectedCharacter,
  } = dash

  const [selectedThreadId, setSelectedThreadId] = useState<number | null>(null)

  useEffect(() => {
    if (!theme) return
    const r = document.documentElement
    r.style.setProperty('--accent',       theme.accent)
    r.style.setProperty('--accent-dark',  theme.accentDark)
    r.style.setProperty('--accent-light', theme.accentLight)
  }, [theme])

  useEffect(() => {
    if (activeView !== 'history') setSelectedThreadId(null)
  }, [activeView])

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
        <div className={styles.content} style={{ backgroundColor: theme.accentLight }}>
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
              currentCharacter={currentCharacter}
              userName={user?.name || ''}
              unlockedAchs={unlockedAchs}
              handleSelectChar={setSelectedCharacter}
            />
          )}
          {activeView === 'history' && (
            <HistoryView
              theme={theme}
              onSelectThread={(id) => setSelectedThreadId(id)}
            />
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

        {activeView === 'history' && (
          <div className={styles.rightPanel}>
            <div className={selectedThreadId !== null ? styles.sceneBlurred : styles.sceneNormal}>
              <ScenePanel
                theme={theme}
                streak={streak}
                currentCharacter={currentCharacter}
                nextAchievement={nextAchievement}
              />
            </div>
            {selectedThreadId !== null && (
              <div className={styles.threadOverlay}>
                <ThreadDetailView
                  threadId={selectedThreadId}
                  theme={theme}
                  onBack={() => setSelectedThreadId(null)}
                />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}