import { useEffect, useState } from 'react'
import { authService } from '../../services/authService'
import type { UserDto } from '../../dtos/authDtos'
import { useAuth } from '../../hooks/useAuth'

export default function Dashboard() {
  const { logout } = useAuth()
  const [user, setUser] = useState<UserDto | null>(null)

  useEffect(() => {
    authService.me().then(setUser).catch(() => {})
  }, [])

  if (!user) return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <p style={{ color: '#8B6EC5' }}>Se încarcă...</p>
    </div>
  )

  return (
    <div style={{ minHeight: '100vh', padding: '2rem', fontFamily: 'Nunito, sans-serif' }}>
      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', marginBottom: '2rem'
        }}>
          <h1 style={{ fontFamily: 'Playfair Display, serif', color: '#2D2640' }}>
            Bun venit, {user.name}!
          </h1>
          <button onClick={logout} style={{
            background: 'none', border: '1.5px solid #D5CEED',
            borderRadius: 8, padding: '0.5rem 1rem',
            color: '#8B6EC5', cursor: 'pointer',
            fontSize: '0.85rem', fontWeight: 500,
          }}>
            Deconectare
          </button>
        </div>
        <p style={{ color: '#9B93B8' }}>Dashboard în construcție.</p>
      </div>
    </div>
  )
}