import styles from './ProgressBar.module.css'

interface Props {
  value:   number
  color?:  string
  height?: number
  label?:  string
}

export default function ProgressBar({ value, color, height = 5, label }: Props) {
  const pct = Math.min(100, Math.max(0, value))
  return (
    <div>
      {label && <p className={styles.label}>{label}</p>}
      <div
        className={styles.track}
        style={{ height }}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={styles.fill}
          style={{ width: `${pct}%`, ...(color ? { background: color } : {}) }}
        />
      </div>
    </div>
  )
}