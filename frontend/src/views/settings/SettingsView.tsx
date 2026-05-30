import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { THEMES, type ThemeId } from '@/constants/themes'
import styles from './SettingsView.module.css'
import { API_URL } from '../../utils/api'

interface Props {
  themeId:       ThemeId
  onThemeChange: (id: ThemeId) => void
}

export default function SettingsView({ themeId, onThemeChange }: Props) {
  const navigate = useNavigate()
  const [name,      setName]      = useState('')
  const [email,     setEmail]     = useState('')
  const [createdAt, setCreatedAt] = useState('')
  const [loading,   setLoading]   = useState(false)
  const [saving,    setSaving]    = useState(false)
  const [error,     setError]     = useState('')
  const [success,   setSuccess]   = useState('')

  useEffect(() => {
    const token = localStorage.getItem('aceso_token') ?? ''
    setLoading(true)
    fetch(`${API_URL}/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.json())
      .then(data => {
        setName(data.name ?? '')
        setEmail(data.email ?? '')
        if (data.created_at) {
          const d = new Date(data.created_at)
          setCreatedAt(d.toLocaleDateString('ro-RO', { day: '2-digit', month: 'long', year: 'numeric' }))
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  async function handleSave(e: React.FormEvent) {
    e.preventDefault()
    setSaving(true); setError(''); setSuccess('')
    try {
      const token = localStorage.getItem('aceso_token') ?? ''
      const res = await fetch('${API_URL}/auth/me', {
        method:  'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body:    JSON.stringify({ name, email }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Eroare la salvare.')
      setSuccess('Profilul a fost actualizat!')
    } catch (err: any) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>

      <div className={styles.themeSection}>
        <h3 className={styles.sectionTitle}>👤 Profil</h3>
        <p className={styles.sectionHint}>
          Vizualizează și actualizează informațiile contului tău.
        </p>

        {loading ? (
          <p style={{ fontSize: 13, color: '#A098BC' }}>Se încarcă...</p>
        ) : (
          <form onSubmit={handleSave} className={styles.profileForm}>
            <div className={styles.profileField}>
              <label className={styles.profileLabel}>Nume complet</label>
              <input
                className={styles.profileInput}
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                required
              />
            </div>
            <div className={styles.profileField}>
              <label className={styles.profileLabel}>Email</label>
              <input
                className={styles.profileInput}
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
              />
            </div>
            {createdAt && (
              <div className={styles.profileField}>
                <label className={styles.profileLabel}>Membru din</label>
                <input
                  className={styles.profileInput}
                  type="text"
                  value={createdAt}
                  disabled
                  style={{ background: '#F5F3FC', color: '#A098BC' }}
                />
              </div>
            )}

            {error   && <p className={styles.profileError}>{error}</p>}
            {success && <p className={styles.profileSuccess}>{success}</p>}

            <div className={styles.profileActions}>
              <button type="submit" className={styles.btnSave} disabled={saving}>
                {saving ? 'Se salvează...' : 'Salvează modificările'}
              </button>
              <button
                type="button"
                className={styles.btnChangePass}
                onClick={() => navigate('/reset-password')}
              >
                🔒 Schimbă parola
              </button>
            </div>
          </form>
        )}
      </div>

      <div className={styles.themeSection}>
        <h3 className={styles.sectionTitle}>🎨 Temă vizuală</h3>
        <p className={styles.sectionHint}>
          Schimbă tema pentru a personaliza culorile și fundalul scenic.
        </p>
        <div className={styles.themePicker}>
          {Object.values(THEMES).map(theme => (
            <button
              key={theme.id}
              type="button"
              className={[
                styles.themeBtn,
                theme.id === themeId ? styles.themeActive : '',
              ].filter(Boolean).join(' ')}
              style={{ '--btn-accent': theme.accent } as React.CSSProperties}
              onClick={() => onThemeChange(theme.id)}
              title={theme.name}
            >
              <span
                className={styles.themeBtnPreview}
                style={{ backgroundImage: `url(${theme.sceneImage})` }}
              />
              <span className={styles.themeBtnLabel}>{theme.label}</span>
            </button>
          ))}
        </div>
      </div>

    </div>
  )
}