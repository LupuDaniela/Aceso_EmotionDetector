import type { ReactNode, HTMLAttributes } from 'react'
import styles from './Card.module.css'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

export default function Card({ children, className = '', ...rest }: CardProps) {
  return (
    <div className={`${styles.card} ${className}`} {...rest}>
      {children}
    </div>
  )
}

interface TitleProps {
  children: ReactNode
  action?:  ReactNode
}

export function CardTitle({ children, action }: TitleProps) {
  return (
    <div className={styles.title}>
      <span>{children}</span>
      {action && <span>{action}</span>}
    </div>
  )
}