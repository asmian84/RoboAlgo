"""
auto_trader — autonomous trading loop.

Scans, risk-gates, and places bracket orders unattended. Paper mode by default;
live trading requires BOTH the ROBOALGO_ALLOW_LIVE=1 env var AND the
`live_confirmed` config flag.

Submodules:
    config     — persisted settings + the live-mode gate
    governor   — pure risk policy (no I/O)
    state      — Redis-backed kill switch, run flag, cycle lock
    audit      — writes and queries the decision trail (never raises)
    reconciler — broker is the source of truth
    loop       — the cycle orchestrator
    health     — liveness and decay reporting
"""
