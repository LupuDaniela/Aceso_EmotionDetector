import { Routes, Route, Navigate } from 'react-router-dom'
import LoginPage    from '@/pages/login/LoginPage'
import RegisterPage from '@/pages/register/RegisterPage'
import Dashboard    from '@/pages/dashboard/Dashboard'
import { useAuth }  from '@/hooks/useAuth'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  const token = localStorage.getItem('aceso_token')

  if (loading) return null
  return (user || token) ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login"         element={<LoginPage />} />
      <Route path="/register"      element={<RegisterPage />} />
      <Route path="/auth/callback" element={<LoginPage />} />
      <Route
        path="/dashboard"
        element={
          <PrivateRoute>
            <Dashboard />
          </PrivateRoute>
        }
      />
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  )
}