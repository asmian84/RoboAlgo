import { createClient } from '@supabase/supabase-js'

const url  = import.meta.env.VITE_SUPABASE_URL
const anon = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!url || !anon) {
  // Fail loud in dev — an empty client silently breaks sign-in with a
  // confusing "not a function" trace when a caller uses v2 methods.
  console.error(
    '[supabase] Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY. ' +
    'Auth calls will fail. Add them to frontend/.env and restart vite.'
  )
}

export const supabase = createClient(url ?? '', anon ?? '', {
  auth: {
    persistSession:  true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
})
