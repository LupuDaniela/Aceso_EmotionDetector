import { useState, useRef, useEffect } from 'react'
import { toDateKey } from '@/utils/date'
import type { AcesoTheme } from '../../constants/themes'
import type { MoodKey, AchievementConfig } from '@/types'
import logoA from '@/assets/logo_aceso_a.png'
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
  theme:             AcesoTheme
  onLogMood:         (dateKey: string, mood: MoodKey) => void
  currentCharacter:  AchievementConfig | null
  unlockedAchs:      AchievementConfig[]
  userName:          string
  handleSelectChar?: (ach: AchievementConfig) => void
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
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/)
  if (parts.length === 1) return parts[0][0]?.toUpperCase() ?? ''
  return (parts[1][0]?.toUpperCase() ?? '') + (parts[0][0]?.toUpperCase() ?? '')
}

async function createThread(token: string): Promise<number | null> {
  try {
    const res = await fetch('/api/chat/thread', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      body:    JSON.stringify({ titlu: null }),
    })
    if (res.ok) {
      const data = await res.json()
      return data.id as number
    }
  } catch {}
  return null
}

export default function ConversationView({
  theme, onLogMood, currentCharacter, unlockedAchs, userName, handleSelectChar,
}: Props) {
  if (!theme) return null

  const [messages,       setMessages]       = useState<Message[]>([])
  const [input,          setInput]          = useState('')
  const [loading,        setLoading]        = useState(false)
  const [showCharPicker, setShowCharPicker] = useState(false)
  const [threadId,       setThreadId]       = useState<number | null>(null)
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
      const token = localStorage.getItem('aceso_token') ?? ''

      let tid = threadId
      if (tid === null) {
        tid = await createThread(token)
        if (tid !== null) setThreadId(tid)
      }

      const res = await fetch('/api/chat/message', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body:    JSON.stringify({ text, thread_id: tid }),
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

  const initials = getInitials(userName)

  return (
    <div
      className={styles.root}
      style={{ backgroundImage: `url(${theme.sceneImage})` }}
    >
      <div className={styles.bgOverlay} />

      <div className={styles.chatWrapper}>

        <div className={styles.header}>
          <p className={styles.motto}>
            Aceso – locul unde mintea ta își găsește libertatea,<br />
            iar sufletul curajul de a se deschide.
          </p>
        </div>

        <div className={styles.messages}>
          {messages.length === 0 && <div className={styles.empty} />}

          {messages.map(msg => (
            <div key={msg.id} className={[styles.msgRow, styles[msg.type]].join(' ')}>
              {msg.type === 'user' ? (
                <>
                  <div className={styles.userBubble} style={{ background: theme.accent }}>
                    {msg.text}
                  </div>
                  <div className={styles.avatar} style={{ background: theme.accent }}>
                    {initials}
                  </div>
                </>
              ) : (
                <>
                  <div className={styles.acesoAvatar}>
                    <img src={logoA} alt="Aceso" />
                  </div>
                  <div className={styles.assistantCard} style={{ borderColor: theme.accentLight }}>
                    {msg.text
                      ? <p className={styles.assistantText}>{msg.text}</p>
                      : <p className={styles.assistantText} style={{ color: '#A098BC', fontStyle: 'italic' }}>
                          Răspunsul empatic nu a putut fi generat.
                        </p>
                    }
                  </div>
                </>
              )}
            </div>
          ))}

          {loading && (
            <div className={[styles.msgRow, styles.assistant].join(' ')}>
              <div className={styles.acesoAvatar}>
                <img src={logoA} alt="Aceso" />
              </div>
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
          <div className={styles.inputRow}>

            {currentCharacter ? (
              <div className={styles.charSlot}>
                {!showCharPicker && (
                  <div className={styles.charTooltip}>Alege un confident</div>
                )}
                <button
                  type="button"
                  className={styles.charBtn}
                  onClick={() => unlockedAchs.length > 1 && setShowCharPicker(p => !p)}
                  title={unlockedAchs.length > 1 ? 'Alege personaj' : currentCharacter.name}
                >
                  <img
                    src={currentCharacter.img}
                    alt={currentCharacter.name}
                    style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </button>

                {showCharPicker && (
                  <div className={styles.charPicker}>
                    {unlockedAchs.map(ach => (
                      <button
                        key={ach.days}
                        type="button"
                        className={[
                          styles.charPickerItem,
                          currentCharacter.days === ach.days ? styles.charPickerActive : '',
                        ].join(' ')}
                        onClick={() => {
                          handleSelectChar?.(ach)
                          setShowCharPicker(false)
                        }}
                      >
                        <img src={ach.img} alt={ach.name} />
                        <span>{ach.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className={styles.charSlotEmpty} />
            )}

            <div className={styles.inputWrapper}>
              <textarea
                className={styles.textarea}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Scrie exact atât cât simți: de la un singur rând, la o poveste întreagă."
                rows={1}
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
    </div>
  )
}