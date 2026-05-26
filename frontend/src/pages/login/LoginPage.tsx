import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { authService } from '../../services/authService'
import { useAuth } from '@/hooks/useAuth'
import styles from './LoginPage.module.css'
import logoAceso from '../../assets/logo_aceso.png'
import authImage from '../../assets/photo_auth.png'

type View = 'login' | 'forgot'

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908C16.658 14.013 17.64 11.705 17.64 9.2z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  )
}

function EyeIcon({ open }: { open: boolean }) {
  return open ? (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
    </svg>
  ) : (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  )
}

export default function LoginPage() {
  const navigate       = useNavigate()
  const { setToken }   = useAuth()           // ← adaugat

  const [view,     setView]     = useState<View>('login')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [showPass, setShowPass] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [success,  setSuccess]  = useState('')
  const [mounted,  setMounted]  = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 60)
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const token  = params.get('token')
    if (token) {
      setToken(token)                        // ← era localStorage.setItem
      window.history.replaceState({}, '', '/auth/callback')
      navigate('/dashboard', { replace: true })
    }
  }, [navigate, setToken])

  function switchView(next: View) {
    setError(''); setSuccess(''); setPassword('')
    setView(next)
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const data = await authService.login({ email, password })
      setToken(data.access_token)            // ← era localStorage.setItem
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  async function handleForgot(e: React.FormEvent) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const data = await authService.forgotPassword({ email })
      setSuccess(data.message)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.bg}>
      <div className={`${styles.card} ${mounted ? styles.visible : ''}`}>
        <div className={styles.left}>

          {view === 'forgot' ? (
            <>
              <button className={styles.backBtn} onClick={() => switchView('login')}>← Înapoi</button>
              <div className={styles.logo}><img src={logoAceso} alt="Aceso" /></div>
              <p className={styles.greeting}>Resetează parola</p>
              {error   && <div className={styles.alertError}>{error}</div>}
              {success && <div className={styles.alertSuccess}>{success}</div>}
              {!success && (
                <form onSubmit={handleForgot}>
                  <div className={styles.field}>
                    <label className={styles.label}>Email</label>
                    <input className={styles.input} type="email" placeholder="tu@exemplu.ro"
                      value={email} onChange={e => setEmail(e.target.value)} required autoFocus />
                  </div>
                  <button className={styles.btnPrimary} disabled={loading}>
                    {loading && <span className={styles.spinner} />}
                    Trimite link de resetare
                  </button>
                </form>
              )}
            </>
          ) : (
            <>
              <div className={styles.logo}><img src={logoAceso} alt="Aceso" /></div>
              <p className={styles.greeting}>Bine ai revenit!</p>

              <div className={styles.tabs}>
                <button className={`${styles.tab} ${styles.active}`}>Autentificare</button>
                <Link to="/register" style={{ flex: 1, textDecoration: 'none' }}>
                  <button className={styles.tab} style={{ width: '100%' }}>Înregistrare</button>
                </Link>
              </div>

              {error && <div className={styles.alertError}>{error}</div>}

              <form onSubmit={handleLogin}>
                <div className={styles.field}>
                  <label className={styles.label}>Email</label>
                  <input className={styles.input} type="email" placeholder="tu@exemplu.ro"
                    value={email} onChange={e => setEmail(e.target.value)} required autoFocus />
                </div>
                <div className={styles.field}>
                  <label className={styles.label}>Parolă</label>
                  <div className={styles.fieldWrap}>
                    <input className={`${styles.input} ${styles.inputWithIcon}`}
                      type={showPass ? 'text' : 'password'} placeholder="••••••••"
                      value={password} onChange={e => setPassword(e.target.value)} required />
                    <button type="button" className={styles.eyeBtn}
                      onClick={() => setShowPass(p => !p)}
                      aria-label={showPass ? 'Ascunde parola' : 'Arată parola'}>
                      <EyeIcon open={showPass} />
                    </button>
                  </div>
                </div>
                <div className={styles.forgotRow}>
                  <button type="button" className={styles.forgotLink} onClick={() => switchView('forgot')}>
                    Ai uitat parola?
                  </button>
                </div>
                <button className={styles.btnPrimary} disabled={loading}>
                  {loading && <span className={styles.spinner} />}
                  Intră în cont
                </button>
              </form>

              <div className={styles.divider}>
                <span className={styles.dividerLine} />
                <span className={styles.dividerText}>sau continuă cu</span>
                <span className={styles.dividerLine} />
              </div>

              <button className={styles.btnGoogle} onClick={authService.googleLogin}>
                <GoogleIcon /> Google
              </button>

              <p className={styles.footer}>
                Nu ai cont?{' '}
                <Link to="/register">
                  <button className={styles.footerBtn}>Înregistrează-te</button>
                </Link>
              </p>
            </>
          )}
        </div>

        <div className={styles.right}>
          <img src={authImage} alt="Aceso vizual" className={styles.rightImg} />
        </div>
      </div>
    </div>
  )
}