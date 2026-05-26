import { useEffect, useState } from 'react'
import type { AcesoTheme } from '../../constants/themes'
import styles from './HistoryView.module.css'

interface ConvItem {
  id:               number
  message:          string
  emotie_dominanta: string
  scor_dominant:    number
  timestamp:        string
}

interface Props {
  theme: AcesoTheme
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

function deriveTitle(message: string): string {
  const trimmed = message.trim()
  if (trimmed.length <= 60) return trimmed
  return trimmed.slice(0, 57) + '...'
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString('ro-RO', {
    day:    '2-digit',
    month:  'long',
    year:   'numeric',
    hour:   '2-digit',
    minute: '2-digit',
  })
}

export default function HistoryView({ theme }: Props) {
  const [items,   setItems]   = useState<ConvItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('aceso_token')
    if (!token) { setLoading(false); return }
    fetch('/api/history?limit=50', {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(data => {
        setItems(Array.isArray(data) ? data : [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className={styles.root}>
      <h1 className={styles.title}>📜 Conversații avute</h1>

      {loading && (
        <div className={styles.empty}>Se încarcă...</div>
      )}

      {!loading && items.length === 0 && (
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>💬</span>
          <p>Nu ai nicio conversație salvată încă.</p>
        </div>
      )}

      <div className={styles.list}>
        {items.map(item => {
          const color = PLUTCHIK[item.emotie_dominanta] ?? '#90A4AE'
          const emoji = EMOTIE_EMOJI[item.emotie_dominanta] ?? '💭'
          return (
            <div key={item.id} className={styles.card}>
              <div className={styles.cardTop}>
                <span
                  className={styles.emotionBadge}
                  style={{ background: color + '22', color, borderColor: color + '55' }}
                >
                  {emoji} {item.emotie_dominanta}
                  <span className={styles.score}>
                    {Math.round(item.scor_dominant * 100)}%
                  </span>
                </span>
                <span className={styles.date}>{formatDate(item.timestamp)}</span>
              </div>
              <p className={styles.convTitle}>{deriveTitle(item.message)}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}