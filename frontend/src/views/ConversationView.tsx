import { useState, useRef, useEffect } from 'react'
import { toDateKey } from '@/utils/date'
import type { AcesoTheme } from '../constants/themes'
import type { MoodKey } from '@/types'
import styles from './ConversationView.module.css'

interface AnalyzeResult {
  emotie_dominanta: string
  scor_dominant:    number
  scoruri:          Record<string, number>
  raspuns_empatic?: string
  diade:            { nume: string; tip: string; scor: number }[]
}

interface Message {
  id:   number
  type: 'user' | 'assistant'
  text: string
}

interface Props {
  theme:     AcesoTheme
  onLogMood: (dateKey: string, mood: MoodKey) => void
}

const EMOTIE_TO_MOOD: Record<string, MoodKey> = {
  Bucurie:    'joy',
  Tristete:   'sadness',
  Frica:      'fear',
  Furie:      'anger',
  Surpriza:   'surprise',
  Incredere:  'trust',
  Anticipare: 'anticipation',
  Dezgust:    'disgust',
  Neutru:     'neutral',
  Iubire:     'love',
}

export default function ConversationView({ theme, onLogMood }: Props) {
  if (!theme) return null

  const [messages,   setMessages]   = useState<Message[]>([])
  const [input,      setInput]      = useState('')
  const [loading,    setLoading]    = useState(false)
  const bottomRef   = useRef<HTMLDivElement>(null)
  const idRef       = useRef(0)
  const loggedToday = useRef(false)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSend() {
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { id: ++idRef.current, type: 'user', text }])
    setInput('')
    setLoading(true)

    try {
      const token = localStorage.getItem('aceso_token')
      const res   = await fetch('/api/analyze', {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ text, salveaza: true }),
      })

      if (!res.ok) throw new Error('Eroare server')
      const data: AnalyzeResult = await res.json()

      if (!loggedToday.current) {
        const mood = EMOTIE_TO_MOOD[data.emotie_dominanta]
        if (mood) {
          const today = new Date()
          today.setHours(0, 0, 0, 0)
          onLogMood(toDateKey(today), mood)
          loggedToday.current = true
        }
      }

      setMessages(prev => [...prev, {
        id:   ++idRef.current,
        type: 'assistant',
        text: data.raspuns_empatic ?? '',
      }])
    } catch {
      setMessages(prev => [...prev, {
        id:   ++idRef.current,
        type: 'assistant',
        text: 'A apărut o eroare. Încearcă din nou.',
      }])
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div
      className={styles.root}
      style={{ backgroundImage: `url(${theme.sceneImage})` }}
    >
      <div className={styles.bgOverlay} />

      <div className={styles.chatWrapper}>
        <div className={styles.header}>
          <h1 className={styles.title}>💬 Conversație nouă</h1>
          <p className={styles.subtitle}>Scrie ce simți — Aceso analizează și răspunde empatic.</p>
        </div>

        <div className={styles.messages}>
          {messages.length === 0 && (
            <div className={styles.empty}>
              <span className={styles.emptyIcon}>🌸</span>
              <p>Începe o conversație scrind cum te simți azi.</p>
            </div>
          )}

          {messages.map(msg => (
            <div key={msg.id} className={[styles.msgRow, styles[msg.type]].join(' ')}>
              {msg.type === 'user' ? (
                <div className={styles.userBubble} style={{ background: theme.accent }}>
                  {msg.text}
                </div>
              ) : (
                <div className={styles.assistantCard} style={{ borderColor: theme.accentLight }}>
                  {msg.text
                    ? <p className={styles.assistantText}>{msg.text}</p>
                    : <p className={styles.assistantText} style={{ color: '#A098BC', fontStyle: 'italic' }}>
                        Răspunsul empatic nu a putut fi generat.
                      </p>
                  }
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className={[styles.msgRow, styles.assistant].join(' ')}>
              <div className={styles.assistantCard} style={{ borderColor: theme.accentLight }}>
                <div className={styles.typing}>
                  <span style={{ background: theme.accent }} />
                  <span style={{ background: theme.accent }} />
                  <span style={{ background: theme.accent }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <textarea
              className={styles.textarea}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Cum te simți azi? Scrie liber..."
              rows={3}
              disabled={loading}
              style={{ borderColor: input ? theme.accent : undefined }}
            />
            <button
              type="button"
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ background: theme.accent }}
            >
              ↑
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}