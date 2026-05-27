import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import styles from '../login/LoginPage.module.css'
import logoAceso from '../../assets/logo_aceso.png'
import authImage  from '../../assets/photo_auth.png'

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

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams()
  const navigate       = useNavigate()
  const token          = searchParams.get('token') ?? ''

  const [password, setPassword] = useState('')
  const [confirm,  setConfirm]  = useState('')
  const [showPass, setShowPass] = useState(false)
  const [showConf, setShowConf] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [success,  setSuccess]  = useState(false)
  const [userInfo, setUserInfo] = useState<{ name: string; email: string } | null>(null)
  const [mounted,  setMounted]  = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 60)
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    if (!token) return
    fetch(`/auth/user-from-token?token=${token}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setUserInfo({ name: data.name, email: data.email }) })
      .catch(() => {})
  }, [token])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (password !== confirm) { setError('Parolele nu coincid.'); return }
    if (password.length < 6)  { setError('Parola trebuie să aibă cel puțin 6 caractere.'); return }
    setLoading(true); setError('')
    try {
      const res = await fetch('/auth/reset-password', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ token, new_password: password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Eroare.')
      setSuccess(true)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.bg}>
      <div className={`${styles.card} ${mounted ? styles.visible : ''}`}>
        <div className={styles.left}>

          <div className={styles.logo}><img src={logoAceso} alt="Aceso" /></div>
          <p className={styles.greeting}>Resetează parola</p>

          {success ? (
            <>
              <div className={styles.alertSuccess}>
                ✅ Parola a fost schimbată cu succes!
              </div>
              <p style={{ fontSize: '0.85rem', color: '#6B6280', textAlign: 'center', margin: '0.5rem 0 1.25rem' }}>
                Te poți autentifica acum cu noua parolă.
              </p>
              <button className={styles.btnPrimary} onClick={() => navigate('/login')}>
                Înapoi la autentificare
              </button>
            </>
          ) : (
            <>
              {userInfo && (
                <>
                  <div className={styles.field}>
                    <label className={styles.label}>Nume</label>
                    <input className={styles.input} type="text" value={userInfo.name}
                      disabled style={{ background: '#F5F3FC', color: '#A098BC' }} />
                  </div>
                  <div className={styles.field}>
                    <label className={styles.label}>Email</label>
                    <input className={styles.input} type="text" value={userInfo.email}
                      disabled style={{ background: '#F5F3FC', color: '#A098BC' }} />
                  </div>
                </>
              )}

              {error && <div className={styles.alertError}>{error}</div>}

              <form onSubmit={handleSubmit}>
                <div className={styles.field}>
                  <label className={styles.label}>Parolă nouă</label>
                  <div className={styles.fieldWrap}>
                    <input className={`${styles.input} ${styles.inputWithIcon}`}
                      type={showPass ? 'text' : 'password'} placeholder="Minim 6 caractere"
                      value={password} onChange={e => setPassword(e.target.value)} required autoFocus />
                    <button type="button" className={styles.eyeBtn}
                      onClick={() => setShowPass(p => !p)}
                      aria-label={showPass ? 'Ascunde parola' : 'Arată parola'}>
                      <EyeIcon open={showPass} />
                    </button>
                  </div>
                </div>

                <div className={styles.field}>
                  <label className={styles.label}>Confirmă parola</label>
                  <div className={styles.fieldWrap}>
                    <input className={`${styles.input} ${styles.inputWithIcon}`}
                      type={showConf ? 'text' : 'password'} placeholder="••••••••"
                      value={confirm} onChange={e => setConfirm(e.target.value)} required />
                    <button type="button" className={styles.eyeBtn}
                      onClick={() => setShowConf(p => !p)}
                      aria-label={showConf ? 'Ascunde parola' : 'Arată parola'}>
                      <EyeIcon open={showConf} />
                    </button>
                  </div>
                </div>

                <button className={styles.btnPrimary} type="submit" disabled={loading}>
                  {loading && <span className={styles.spinner} />}
                  Salvează parola
                </button>
              </form>

              <p className={styles.footer}>
                <button className={styles.footerBtn} onClick={() => navigate('/login')}>
                  ← Înapoi la autentificare
                </button>
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