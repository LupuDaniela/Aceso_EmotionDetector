import { useEffect, useState } from 'react'
import MoodCalendar        from '@/components/calendar/MoodCalendar'
import Card, { CardTitle } from '@/components/card/Card'
import { MONTHS_RO }       from '@/constants/calendar'
import { ACHIEVEMENTS }    from '@/constants/achievements'
import type { CalendarProps } from '@/types'
import styles from './StatsView.module.css'

interface Props {
  streak:        number
  unlockedCount: number
  calendar:      CalendarProps & { currentYear: number; currentMonth: number }
}

interface ApiStats {
  total:       number
  distributie: { emotie: string; count: number }[]
  diade:       { diada: string; count: number }[]
}

const PLUTCHIK: Record<string, string> = {
  Bucurie:    '#FDD835',
  Tristete:   '#42A5F5',
  Frica:      '#AB47BC',
  Furie:      '#EF5350',
  Surpriza:   '#FF7043',
  Incredere:  '#26A69A',
  Anticipare: '#FFA726',
  Dezgust:    '#66BB6A',
  Neutru:     '#90A4AE',
  Iubire:     '#F48FB1',
}

const DIADE_COLORS: Record<string, string> = {
  Iubire:         '#F48FB1',
  Supunere:       '#CE93D8',
  Teama:          '#9575CD',
  Dezamagire:     '#7986CB',
  Remuscare:      '#64B5F6',
  Dispret:        '#4DB6AC',
  Agresivitate:   '#FF8A65',
  Optimism:       '#FFD54F',
  Vinovatie:      '#A5D6A7',
  Curiozitate:    '#80DEEA',
  Disperare:      '#B39DDB',
  Rusine:         '#F48FB1',
  Invidie:        '#EF9A9A',
  Cinism:         '#80CBC4',
  Mandrie:        '#FFCC02',
  Speranta:       '#AED581',
  Incantare:      '#FFF176',
  Sentimentalism: '#90CAF9',
  Pudoare:        '#CE93D8',
  Indignare:      '#FFAB91',
  Pesimism:       '#B0BEC5',
  Morbiditate:    '#A5D6A7',
  Dominanta:      '#EF5350',
  Anxietate:      '#9575CD',
}

const EMOTIE_EMOJI: Record<string, string> = {
  Bucurie:    '😄',
  Tristete:   '😢',
  Frica:      '😨',
  Furie:      '😡',
  Surpriza:   '😲',
  Incredere:  '🤝',
  Anticipare: '🌟',
  Dezgust:    '🤢',
  Neutru:     '😐',
  Iubire:     '🥰',
}

export default function StatsView({ streak, unlockedCount, calendar }: Props) {
  const [apiStats, setApiStats] = useState<ApiStats | null>(null)

  useEffect(() => {
    const token = localStorage.getItem('aceso_token')
    fetch('/api/stats', {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.json())
      .then(setApiStats)
      .catch(() => {})
  }, [])

  const generalStats = [
    { val: streak,                                    lbl: 'Streak curent (zile)' },
    { val: `${unlockedCount}/${ACHIEVEMENTS.length}`, lbl: 'Avataruri deblocate'  },
    { val: Object.keys(calendar.moodLog).length,      lbl: 'Zile notate total'    },
    { val: apiStats?.total ?? '—',                    lbl: 'Conversații totale'   },
  ]

  const maxEmotie = apiStats?.distributie[0]?.count ?? 1
  const maxDiada  = apiStats?.diade[0]?.count ?? 1

  return (
    <>
      <Card>
        <CardTitle>📊 Statistici generale</CardTitle>
        <div className={styles.grid}>
          {generalStats.map(s => (
            <div key={s.lbl} className={styles.statCard}>
              <div className={styles.val}>{s.val}</div>
              <div className={styles.lbl}>{s.lbl}</div>
            </div>
          ))}
        </div>
      </Card>

      {apiStats && apiStats.distributie.length > 0 && (
        <Card>
          <CardTitle>🎭 Emoții dominante (simple)</CardTitle>
          <div className={styles.chartList}>
            {apiStats.distributie.map(({ emotie, count }) => {
              const color = PLUTCHIK[emotie] ?? '#90A4AE'
              const emoji = EMOTIE_EMOJI[emotie] ?? '💭'
              const pct   = Math.round((count / maxEmotie) * 100)
              return (
                <div key={emotie} className={styles.chartRow}>
                  <span className={styles.chartLabel}>{emoji} {emotie}</span>
                  <div className={styles.chartBarBg}>
                    <div
                      className={styles.chartBarFill}
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                  <span className={styles.chartCount}>{count}</span>
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {apiStats && apiStats.diade.length > 0 && (
        <Card>
          <CardTitle>🔗 Emoții complexe (diade)</CardTitle>
          <div className={styles.chartList}>
            {apiStats.diade.slice(0, 10).map(({ diada, count }) => {
              const color = DIADE_COLORS[diada] ?? '#B39DDB'
              const pct   = Math.round((count / maxDiada) * 100)
              return (
                <div key={diada} className={styles.chartRow}>
                  <span className={styles.chartLabel}>{diada}</span>
                  <div className={styles.chartBarBg}>
                    <div
                      className={styles.chartBarFill}
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                  <span className={styles.chartCount}>{count}</span>
                </div>
              )
            })}
          </div>
          <p className={styles.chartNote}>
            Diadele sunt combinații de emoții primare detectate în conversații.
          </p>
        </Card>
      )}

      <Card>
        <CardTitle>
          📅 {MONTHS_RO[calendar.currentMonth]} {calendar.currentYear}
        </CardTitle>
        <MoodCalendar {...calendar} showLegend light />
      </Card>
    </>
  )
}