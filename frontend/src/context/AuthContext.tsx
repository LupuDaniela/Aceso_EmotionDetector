import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { authService } from '@/services/authService'
import type { UserDto } from '@/dtos/authDtos'

interface AuthContextType {
  user:     UserDto | null
  loading:  boolean
  logout:   () => void
  setToken: (token: string) => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,    setUser]    = useState<UserDto | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('aceso_token')
    if (!token) { setLoading(false); return }
    authService.me()
      .then(data => setUser(data))
      .catch(() => {
        localStorage.removeItem('aceso_token')
        setUser(null)
      })
      .finally(() => setLoading(false))
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('aceso_token')
    setUser(null)
  }, [])

  const setToken = useCallback((token: string) => {
    localStorage.setItem('aceso_token', token)
    setLoading(true)
    authService.me()
      .then(data => setUser(data))
      .catch(() => {
        localStorage.removeItem('aceso_token')
        setUser(null)
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, logout, setToken }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuthContext() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuthContext must be used inside AuthProvider')
  return ctx
}