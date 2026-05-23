import { useState } from 'react'
import { MOOD_CONFIG, MONTHS_RO, WEEKDAYS_SHORT } from '@/constants/calendar'
import { toDateKey, getMonthStartOffset, getDaysInMonth, isFutureDay, isTodayDay } from '@/utils/date'
import MoodPicker from '../mood/MoodPicker'
import type { CalendarProps, MoodKey } from '@/types'
import styles from '../calendar/ModdCalendar.module.css'

interface Props extends CalendarProps {
  showLegend?: boolean
}

export default function MoodCalendar({
  moodLog, currentYear, currentMonth,
  canGoNext, onLogMood, onRemoveMood,
  onPrevMonth, onNextMonth, onToday,
  showLegend = false,
}: Props) {
  const [pickerKey, setPickerKey] = useState<string | null>(null)

  const totalDays   = getDaysInMonth(currentYear, currentMonth)
  const startOffset = getMonthStartOffset(currentYear, currentMonth)

  const now            = new Date()
  const isCurrentMonth =
    currentYear  === now.getFullYear() &&
    currentMonth === now.getMonth()

  function handleDayClick(day: number) {
    if (isFutureDay(currentYear, currentMonth, day)) return
    setPickerKey(toDateKey(new Date(currentYear, currentMonth, day)))
  }

  function handleSelect(mood: MoodKey) {
    if (!pickerKey) return
    onLogMood(pickerKey, mood)
    setPickerKey(null)
  }

  function handleRemove() {
    if (!pickerKey) return
    onRemoveMood(pickerKey)
    setPickerKey(null)
  }

  return (
    <div className={styles.root}>
      <div className={styles.nav}>
        <button
          type="button"
          className={styles.navBtn}
          onClick={onPrevMonth}
          aria-label="Luna anterioară"
        >
          ‹
        </button>

        <div className={styles.monthLabel}>
          <span>{MONTHS_RO[currentMonth]} {currentYear}</span>
          {!isCurrentMonth && (
            <button type="button" className={styles.todayBtn} onClick={onToday}>
              Azi
            </button>
          )}
        </div>

        <button
          type="button"
          className={styles.navBtn}
          onClick={onNextMonth}
          disabled={!canGoNext}
          aria-label="Luna următoare"
        >
          ›
        </button>
      </div>

      <div className={styles.grid}>
        {WEEKDAYS_SHORT.map(d => (
          <div key={d} className={styles.weekday}>{d}</div>
        ))}

        {Array.from({ length: startOffset }, (_, i) => (
          <div key={`gap-${i}`} aria-hidden="true" />
        ))}

        {Array.from({ length: totalDays }, (_, i) => {
          const day     = i + 1
          const future  = isFutureDay(currentYear, currentMonth, day)
          const isToday = isTodayDay(currentYear, currentMonth, day)
          const dateKey = toDateKey(new Date(currentYear, currentMonth, day))
          const mood    = moodLog[dateKey] as MoodKey | undefined
          const cfg     = mood ? MOOD_CONFIG[mood] : undefined

          return (
            <div
              key={day}
              className={[
                styles.day,
                future  ? styles.future  : styles.past,
                isToday ? styles.today   : '',
                mood    ? styles.hasMood : '',
              ].filter(Boolean).join(' ')}
              style={cfg ? { background: cfg.color } : undefined}
              onClick={() => handleDayClick(day)}
              title={future ? undefined : cfg ? cfg.label : 'Click pentru a nota dispoziția'}
              aria-label={cfg ? `${day} ${MONTHS_RO[currentMonth]} — ${cfg.label}` : String(day)}
            >
              {mood
                ? <span className={styles.emoji}>{cfg!.emoji}</span>
                : <span className={styles.num}>{day}</span>
              }
              {!mood && !future && (
                <span className={styles.addHint} aria-hidden="true">+</span>
              )}
            </div>
          )
        })}
      </div>

      {showLegend && (
        <div className={styles.legend} role="list" aria-label="Legendă dispoziții">
          {(Object.entries(MOOD_CONFIG) as [MoodKey, typeof MOOD_CONFIG[MoodKey]][]).map(
            ([key, { emoji, label, color }]) => (
              <div key={key} className={styles.legendItem} role="listitem">
                <span
                  className={styles.legendDot}
                  style={{ background: color }}
                  aria-hidden="true"
                >
                  {emoji}
                </span>
                <span className={styles.legendLabel}>{label}</span>
              </div>
            )
          )}
        </div>
      )}

      {pickerKey && (
        <MoodPicker
          dateKey={pickerKey}
          currentMood={moodLog[pickerKey] as MoodKey | undefined}
          onSelect={handleSelect}
          onRemove={handleRemove}
          onClose={() => setPickerKey(null)}
        />
      )}
    </div>
  )
}