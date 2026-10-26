import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import type { Session, User } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'

interface AuthContextValue {
  session: Session | null
  user:    User    | null
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue>({
  session: null,
  user:    null,
  loading: true,
  signOut: async () => {},
})

// Dev-only escape hatch — when Supabase is paused/unreachable, set
// VITE_DEV_SKIP_AUTH=1 in frontend/.env to mint a local session and
// bypass the sign-in page entirely. NEVER set this in production.
const DEV_SKIP_AUTH = import.meta.env.VITE_DEV_SKIP_AUTH === '1'

const _DEV_SESSION = {
  access_token:  'dev',
  refresh_token: 'dev',
  expires_in:    3600,
  expires_at:    Math.floor(Date.now() / 1000) + 3600,
  token_type:    'bearer',
  user: {
    id:              'dev-user',
    email:           'dev@localhost',
    aud:             'authenticated',
    role:            'authenticated',
    app_metadata:    {},
    user_metadata:   {},
    created_at:      new Date().toISOString(),
  },
} as unknown as Session

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(DEV_SKIP_AUTH ? _DEV_SESSION : null)
  const [loading, setLoading] = useState(!DEV_SKIP_AUTH)

  useEffect(() => {
    if (DEV_SKIP_AUTH) {
      console.warn('[auth] VITE_DEV_SKIP_AUTH=1 — bypassing Supabase, using local dev session')
      return
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setLoading(false)
    })

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      setLoading(false)
    })

    return () => subscription.unsubscribe()
  }, [])

  const signOut = async () => {
    if (DEV_SKIP_AUTH) {
      setSession(null)
      return
    }
    await supabase.auth.signOut()
  }

  return (
    <AuthContext.Provider value={{ session, user: session?.user ?? null, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
