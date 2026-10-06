import { createContext, useContext, type ReactNode } from 'react'

// Supabase temporarily disabled — auth is bypassed locally.
// To reconnect: restore the original Supabase-backed implementation.
const FAKE_USER = { email: 'local@dev' } as { email: string }
const FAKE_SESSION = { user: FAKE_USER } as { user: typeof FAKE_USER }

interface AuthContextValue {
  session: typeof FAKE_SESSION | null
  user: typeof FAKE_USER | null
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue>({
  session: FAKE_SESSION,
  user: FAKE_USER,
  loading: false,
  signOut: async () => {},
})

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <AuthContext.Provider
      value={{
        session: FAKE_SESSION,
        user: FAKE_USER,
        loading: false,
        signOut: async () => {},
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
