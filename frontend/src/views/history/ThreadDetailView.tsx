import { useEffect, useState } from 'react'
import type { AcesoTheme } from '../../constants/themes'
import logoA from '@/assets/logo_aceso_a.png'
import styles from './ThreadDetailView.module.css'

interface ThreadMessage {
  id:               number
  message:          string
  emotie_dominanta: string
  scor_dominant:    number
  raspuns_empatic:  string | null
  timestamp:        string
}

interface Props {
  threadId: number
  theme:    AcesoTheme
  onBack:   () => void
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
}

const EMOTIE_EMOJI: Record<string, string> = {
  Bucurie: '😄', Tristete: '😢', Frica: '😨', Furie: '😡',
  Surpriza: '😲', Incredere: '🤝', Anticipare: '🌟',
  Dezgust: '🤢', Neutru: '😐',
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('ro-RO', { hour: '2-digit', minute: '2-digit' })
}

export default function ThreadDetailView({ threadId, theme, onBack }: Props) {
  const [messages, setMessages] = useState<ThreadMessage[]>([])
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    setLoading(true)
    setMessages([])
    const token = localStorage.getItem('aceso_token')
    if (!token) { setLoading(false); return }
    fetch(`/api/chat/thread/${threadId}/messages`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then(data => { setMessages(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [threadId])

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <button className={styles.backBtn} onClick={onBack} style={{ color: theme.accent }}>
          ← Înapoi
        </button>
      </div>

      <div className={styles.messages}>
        {loading && <div className={styles.hint}>Se încarcă...</div>}
        {!loading && messages.length === 0 && (
          <div className={styles.hint}>Niciun mesaj în această conversație.</div>
        )}

        {messages.map(msg => {
          const color = PLUTCHIK[msg.emotie_dominanta] ?? '#90A4AE'
          const emoji = EMOTIE_EMOJI[msg.emotie_dominanta] ?? '💭'
          return (
            <div key={msg.id} className={styles.pair}>
              <div className={styles.userRow}>
                <div className={styles.userBubble} style={{ background: theme.accent }}>
                  {msg.message}
                </div>
                <span
                  className={styles.emotionBadge}
                  style={{ background: color + '22', color, borderColor: color + '55' }}
                >
                  {emoji} {msg.emotie_dominanta}
                  <span className={styles.score}>{Math.round(msg.scor_dominant * 100)}%</span>
                </span>
                <span className={styles.time}>{formatTime(msg.timestamp)}</span>
              </div>

              {msg.raspuns_empatic && (
                <div className={styles.assistantRow}>
                  <div className={styles.acesoAvatar}>
                    <img src={logoA} alt="Aceso" />
                  </div>
                  <div className={styles.assistantBubble} style={{ borderColor: theme.accentLight }}>
                    {msg.raspuns_empatic}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}