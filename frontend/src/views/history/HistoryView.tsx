// import { useEffect, useState } from 'react'
// import type { AcesoTheme } from '../../constants/themes'
// import ThreadDetailView from './ThreadDetailView'
// import styles from './HistoryView.module.css'

// interface ThreadItem {
//   id:            number
//   titlu:         string
//   actualizat_la: string
//   last_message:  string | null
//   last_emotie:   string | null
//   last_diade:    string[]
// }

// interface Props {
//   theme: AcesoTheme
// }

// const PLUTCHIK: Record<string, string> = {
//   Bucurie:    '#FDD835',
//   Tristete:   '#42A5F5',
//   Frica:      '#AB47BC',
//   Furie:      '#EF5350',
//   Surpriza:   '#FF7043',
//   Incredere:  '#26A69A',
//   Anticipare: '#FFA726',
//   Dezgust:    '#66BB6A',
//   Neutru:     '#90A4AE',
// }

// const EMOTIE_EMOJI: Record<string, string> = {
//   Bucurie: '😄', Tristete: '😢', Frica: '😨', Furie: '😡',
//   Surpriza: '😲', Incredere: '🤝', Anticipare: '🌟',
//   Dezgust: '🤢', Neutru: '😐',
// }

// function formatDate(iso: string): string {
//   return new Date(iso).toLocaleDateString('ro-RO', {
//     day: '2-digit', month: 'long', year: 'numeric',
//     hour: '2-digit', minute: '2-digit',
//   })
// }

// export default function HistoryView({ theme }: Props) {
//   const [items,      setItems]      = useState<ThreadItem[]>([])
//   const [loading,    setLoading]    = useState(true)
//   const [selectedId, setSelectedId] = useState<number | null>(null)

//   useEffect(() => {
//     const token = localStorage.getItem('aceso_token')
//     if (!token) { setLoading(false); return }
//     fetch('/api/chat/threads', { headers: { 'Authorization': `Bearer ${token}` } })
//       .then(r => r.ok ? r.json() : [])
//       .then(data => { setItems(Array.isArray(data) ? data : []); setLoading(false) })
//       .catch(() => setLoading(false))
//   }, [])

//   if (selectedId !== null) {
//     return (
//       <ThreadDetailView
//         threadId={selectedId}
//         theme={theme}
//         onBack={() => setSelectedId(null)}
//       />
//     )
//   }

//   return (
//     <div className={styles.root}>
//       <h1 className={styles.title}>📜 Conversații avute</h1>

//       {loading && <div className={styles.empty}>Se încarcă...</div>}

//       {!loading && items.length === 0 && (
//         <div className={styles.empty}>
//           <span className={styles.emptyIcon}>💬</span>
//           <p>Nu ai nicio conversație salvată încă.</p>
//         </div>
//       )}

//       <div className={styles.list}>
//         {items.map(item => {
//           const color = PLUTCHIK[item.last_emotie ?? ''] ?? theme.accent
//           const emoji = EMOTIE_EMOJI[item.last_emotie ?? ''] ?? '💭'
//           const diada = item.last_diade?.[0] ?? null

//           return (
//             <div
//               key={item.id}
//               className={styles.card}
//               onClick={() => setSelectedId(item.id)}
//               style={{ cursor: 'pointer' }}
//             >
//               <div className={styles.cardTop}>
//                 <div className={styles.badges}>
//                   {item.last_emotie && (
//                     <span
//                       className={styles.emotionBadge}
//                       style={{ background: color + '22', color, borderColor: color + '55' }}
//                     >
//                       {emoji} {item.last_emotie}
//                     </span>
//                   )}
//                   {diada && (
//                     <span
//                       className={styles.diadaBadge}
//                       style={{
//                         background:  theme.accentLight,
//                         color:       theme.accent,
//                         borderColor: theme.accent + '44',
//                       }}
//                     >
//                       🔗 {diada}
//                     </span>
//                   )}
//                 </div>
//                 <span className={styles.date}>{formatDate(item.actualizat_la)}</span>
//               </div>
//               {item.last_message && (
//                 <p className={styles.convTitle}>
//                   {item.last_message.length > 80
//                     ? item.last_message.slice(0, 77) + '...'
//                     : item.last_message}
//                 </p>
//               )}
//             </div>
//           )
//         })}
//       </div>
//     </div>
//   )
// }

import { useEffect, useState } from 'react'
import type { AcesoTheme } from '../../constants/themes'
import styles from './HistoryView.module.css'

interface ThreadItem {
  id:            number
  titlu:         string
  actualizat_la: string
  last_message:  string | null
  last_emotie:   string | null
  last_diade:    string[]
}

interface Props {
  theme: AcesoTheme
  onSelectThread: (id: number) => void
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

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('ro-RO', {
    day: '2-digit', month: 'long', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export default function HistoryView({ theme, onSelectThread }: Props) {
  const [items,      setItems]      = useState<ThreadItem[]>([])
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('aceso_token')
    if (!token) { setLoading(false); return }
    fetch('/api/chat/threads', { headers: { 'Authorization': `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : [])
      .then(data => { setItems(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className={styles.root}>
      <h1 className={styles.title}>📜 Conversații avute</h1>

      {loading && <div className={styles.empty}>Se încarcă...</div>}

      {!loading && items.length === 0 && (
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>💬</span>
          <p>Nu ai nicio conversație salvată încă.</p>
        </div>
      )}

      <div className={styles.list}>
        {items.map(item => {
          const color = PLUTCHIK[item.last_emotie ?? ''] ?? theme.accent
          const emoji = EMOTIE_EMOJI[item.last_emotie ?? ''] ?? '💭'
          const diada = item.last_diade?.[0] ?? null

          return (
            <div
              key={item.id}
              className={styles.card}
              onClick={() => onSelectThread(item.id)}
              style={{ cursor: 'pointer' }}
            >
              <div className={styles.cardTop}>
                <div className={styles.badges}>
                  {item.last_emotie && (
                    <span
                      className={styles.emotionBadge}
                      style={{ background: color + '22', color, borderColor: color + '55' }}
                    >
                      {emoji} {item.last_emotie}
                    </span>
                  )}
                  {diada && (
                    <span
                      className={styles.diadaBadge}
                      style={{
                        background:  theme.accentLight,
                        color:       theme.accent,
                        borderColor: theme.accent + '44',
                      }}
                    >
                      🔗 {diada}
                    </span>
                  )}
                </div>
                <span className={styles.date}>{formatDate(item.actualizat_la)}</span>
              </div>
              {item.last_message && (
                <p className={styles.convTitle}>
                  {item.last_message.length > 80
                    ? item.last_message.slice(0, 77) + '...'
                    : item.last_message}
                </p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}