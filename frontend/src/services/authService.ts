import type {
  LoginDto, RegisterDto, ForgotPasswordDto,
  ResetPasswordDto, AuthResponseDto, UserDto,
} from '@/dtos/authDtos'

const API = (import.meta as any).env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem('aceso_token')
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers as Record<string, string> ?? {}),
  }
  const res  = await fetch(`${API}${path}`, { ...options, headers })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? 'A apărut o eroare.')
  return data as T
}

export const authService = {
  login:          (dto: LoginDto) =>
    request<AuthResponseDto>('/auth/login',           { method: 'POST', body: JSON.stringify(dto) }),
  register:       (dto: RegisterDto) =>
    request<AuthResponseDto>('/auth/register',         { method: 'POST', body: JSON.stringify(dto) }),
  forgotPassword: (dto: ForgotPasswordDto) =>
    request<{ message: string }>('/auth/forgot-password', { method: 'POST', body: JSON.stringify(dto) }),
  resetPassword:  (dto: ResetPasswordDto) =>
    request<{ message: string }>('/auth/reset-password',  { method: 'POST', body: JSON.stringify(dto) }),
  me: () =>
    request<UserDto>('/auth/me'),
  googleLogin: () => {
    window.location.href = `${API}/auth/google/login`
  },
}