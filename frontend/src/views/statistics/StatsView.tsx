import { useEffect, useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import Card, { CardTitle } from '@/components/card/Card'
import { ACHIEVEMENTS }    from '@/constants/achievements'
import type { CalendarProps } from '@/types'
import styles from './StatsView.module.css'
import { API_URL } from '../../utils/api'

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

interface TimelinePoint {
  date:             string
  total:            number
  emotie_dominanta: string
}

const PLUTCHIK: Record<string, string> = {
  Bucurie:    '#FDD835',
  Tristete:   '#02345d',
  Frica:      '#AB47BC',
  Furie:      '#EF5350',
  Surpriza:   '#FF7043',
  Incredere:  '#26A69A',
  Anticipare: '#FFA726',
  Dezgust:    '#66BB6A',
  Neutru:     '#786842',
}

const DIADE_COLORS: Record<string, string> = {
  Iubire: '#F48FB1', Supunere: '#CE93D8', Teama: '#9575CD',
  Dezamagire: '#7986CB', Remuscare: '#64B5F6', Dispret: '#4DB6AC',
  Agresivitate: '#FF8A65', Optimism: '#FFD54F', Vinovatie: '#A5D6A7',
  Curiozitate: '#80DEEA', Disperare: '#B39DDB', Rusine: '#F48FB1',
  Invidie: '#EF9A9A', Cinism: '#80CBC4', Mandrie: '#FFCC02',
  Speranta: '#AED581', Incantare: '#FFF176', Sentimentalism: '#90CAF9',
  Pudoare: '#CE93D8', Indignare: '#FFAB91', Pesimism: '#B0BEC5',
  Morbiditate: '#A5D6A7', Dominanta: '#EF5350', Anxietate: '#9575CD',
}

const EMOTIE_EMOJI: Record<string, string> = {
  Bucurie: '😄', Tristete: '😢', Frica: '😨', Furie: '😡',
  Surpriza: '😲', Incredere: '🤝', Anticipare: '🌟',
  Dezgust: '🤢', Neutru: '😐'
}

const PLUTCHIK_ORDER = [
  'Bucurie', 'Anticipare', 'Incredere', 'Frica',
  'Surpriza', 'Tristete', 'Dezgust', 'Furie',
]

function formatDateShort(iso: string): string {
  const d = new Date(iso)
  return `${d.getDate()} ${['Ian','Feb','Mar','Apr','Mai','Iun','Iul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()]}`
}

export default function StatsView({ streak, unlockedCount, calendar }: Props) {
  const [apiStats, setApiStats] = useState<ApiStats | null>(null)
  const [timeline, setTimeline] = useState<TimelinePoint[]>([])

  useEffect(() => {
    const token = localStorage.getItem('aceso_token')
    const h = { 'Authorization': `Bearer ${token}` }
    fetch(`${API_URL}/api/stats`,          { headers: h }).then(r => r.json()).then(setApiStats).catch(() => {})
    fetch(`${API_URL}/api/stats/timeline`, { headers: h }).then(r => r.json()).then(setTimeline).catch(() => {})
  }, [])

  const generalStats = [
    { val: streak,                                    lbl: 'Streak curent (zile)' },
    { val: `${unlockedCount}/${ACHIEVEMENTS.length}`, lbl: 'Avataruri deblocate'  },
    { val: Object.keys(calendar.moodLog).length,      lbl: 'Zile notate total'    },
    { val: apiStats?.total ?? '—',                    lbl: 'Conversații totale'   },
  ]

  const radarData = PLUTCHIK_ORDER.map(emotie => {
    const found = apiStats?.distributie.find(d => d.emotie === emotie)
    return { emotie, count: found?.count ?? 0 }
  })

  const maxEmotie = apiStats?.distributie[0]?.count ?? 1
  const maxDiada  = apiStats?.diade[0]?.count ?? 1

  const accentColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--accent').trim() || '#8B6EC5'

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
          <CardTitle>🕸️ Profil emoțional</CardTitle>
          <div className={styles.radarWrap}>
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <PolarGrid stroke="rgba(139,110,197,0.15)" />
                <PolarAngleAxis
                  dataKey="emotie"
                  tick={{ fontSize: 12, fill: '#6B6280', fontWeight: 600 }}
                />
                <Radar
                  dataKey="count"
                  stroke={accentColor}
                  fill={accentColor}
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {timeline.length > 1 && (
        <Card>
          <CardTitle>📈 Activitate zilnică (ultimele 30 zile)</CardTitle>
          <div className={styles.areaWrap}>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={timeline} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={accentColor} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={accentColor} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,110,197,0.10)" />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatDateShort}
                  tick={{ fontSize: 10, fill: '#A098BC' }}
                  interval="preserveStartEnd"
                />
                <YAxis tick={{ fontSize: 10, fill: '#A098BC' }} allowDecimals={false} />
                <Tooltip
                  formatter={(val) => [`${val ?? 0} conversații`, 'Total']}
                  labelFormatter={(label: string) => formatDateShort(label)}
                  contentStyle={{
                    background: '#fff',
                    border: '1px solid rgba(139,110,197,0.2)',
                    borderRadius: 10,
                    fontSize: 12,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="total"
                  stroke={accentColor}
                  strokeWidth={2}
                  fill="url(#areaGrad)"
                  dot={{ r: 3, fill: accentColor, strokeWidth: 0 }}
                  activeDot={{ r: 5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

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
                    <div className={styles.chartBarFill} style={{ width: `${pct}%`, background: color }} />
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
            {apiStats.diade.map(({ diada, count }) => {
              const color = DIADE_COLORS[diada] ?? '#B39DDB'
              const pct   = Math.round((count / maxDiada) * 100)
              return (
                <div key={diada} className={styles.chartRow}>
                  <span className={styles.chartLabel}>{diada}</span>
                  <div className={styles.chartBarBg}>
                    <div className={styles.chartBarFill} style={{ width: `${pct}%`, background: color }} />
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
    </>
  )
}