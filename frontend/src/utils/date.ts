export function toDateKey(date: Date): string {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

export function getMonthStartOffset(year: number, month: number): number {
  const day = new Date(year, month, 1).getDay()
  return day === 0 ? 6 : day - 1
}

export function getDaysInMonth(year: number, month: number): number {
  return new Date(year, month + 1, 0).getDate()
}

export function isFutureDay(year: number, month: number, day: number): boolean {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return new Date(year, month, day) > today
}

export function isTodayDay(year: number, month: number, day: number): boolean {
  const t = new Date()
  t.setHours(0, 0, 0, 0)
  return (
    t.getFullYear() === year &&
    t.getMonth()    === month &&
    t.getDate()     === day
  )
}