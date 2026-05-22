export interface LoginDto {
  email:    string
  password: string
}

export interface RegisterDto {
  email:    string
  name:     string
  password: string
}

export interface ForgotPasswordDto {
  email: string
}

export interface ResetPasswordDto {
  token:        string
  new_password: string
}

export interface AuthResponseDto {
  access_token: string
}

export interface UserDto {
  id:         number
  email:      string
  name:       string
  created_at: string | null
}