import { useState } from 'react'
import { supabase } from '../lib/supabase'

type Mode = 'signin' | 'signup' | 'magic'

export default function LoginPage() {
  const [mode,     setMode]     = useState<Mode>('signin')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [busy,     setBusy]     = useState(false)
  const [message,  setMessage]  = useState<{ text: string; ok: boolean } | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setMessage(null)

    try {
      if (mode === 'magic') {
        const { error } = await supabase.auth.signInWithOtp({
          email,
          options: { emailRedirectTo: window.location.origin },
        })
        if (error) throw error
        setMessage({ text: '✉️ Magic link sent — check your inbox.', ok: true })
      } else if (mode === 'signup') {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: { emailRedirectTo: window.location.origin },
        })
        if (error) throw error
        setMessage({ text: '✅ Account created — check your email to confirm.', ok: true })
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        // AuthContext listener will pick up the session and redirect
      }
    } catch (err: unknown) {
      setMessage({ text: (err as Error).message ?? 'Authentication failed', ok: false })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="text-center mb-8">
          <div className="text-3xl font-black tracking-widest text-emerald-400 mb-1">ROBOALGO</div>
          <div className="text-xs text-gray-600 tracking-widest uppercase">Cycle Trading Intelligence</div>
        </div>

        {/* Card */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl">

          {/* Mode tabs */}
          <div className="flex gap-1 bg-gray-800 rounded-lg p-1 mb-6">
            {(['signin', 'signup', 'magic'] as Mode[]).map(m => (
              <button
                key={m}
                onClick={() => { setMode(m); setMessage(null) }}
                className={`flex-1 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  mode === m
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {m === 'signin' ? 'Sign In' : m === 'signup' ? 'Sign Up' : 'Magic Link'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {/* Email */}
            <div>
              <label className="block text-[11px] text-gray-500 mb-1 uppercase tracking-wide">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-emerald-500 transition-colors"
              />
            </div>

            {/* Password — hidden for magic link */}
            {mode !== 'magic' && (
              <div>
                <label className="block text-[11px] text-gray-500 mb-1 uppercase tracking-wide">Password</label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  minLength={6}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-emerald-500 transition-colors"
                />
              </div>
            )}

            {/* Status message */}
            {message && (
              <div className={`text-xs rounded-lg px-3 py-2 ${
                message.ok
                  ? 'bg-emerald-900/30 text-emerald-300 border border-emerald-800/50'
                  : 'bg-red-900/30 text-red-300 border border-red-800/50'
              }`}>
                {message.text}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={busy}
              className="w-full py-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-bold text-sm transition-colors mt-1"
            >
              {busy
                ? 'Please wait…'
                : mode === 'signin' ? 'Sign In'
                : mode === 'signup' ? 'Create Account'
                : 'Send Magic Link'}
            </button>
          </form>

          {/* Toggle hint */}
          <p className="text-center text-[11px] text-gray-600 mt-5">
            {mode === 'signin'
              ? <>No account? <button onClick={() => setMode('signup')} className="text-emerald-500 hover:text-emerald-300">Sign up</button></>
              : <>Have an account? <button onClick={() => setMode('signin')} className="text-emerald-500 hover:text-emerald-300">Sign in</button></>}
          </p>
        </div>

        <p className="text-center text-[10px] text-gray-700 mt-6">
          Powered by RoboAlgo v3 · Private Access Only
        </p>
      </div>
    </div>
  )
}
