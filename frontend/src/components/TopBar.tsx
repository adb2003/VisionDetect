import { Cpu, RefreshCw, Settings as SettingsIcon } from 'lucide-react'

import { classNames } from '../lib/format'
import type { Health } from '../types'
import { Spinner } from './ui'

/**
 * Connection and model state, always visible.
 *
 * The two indicators are deliberately separate: the backend can be reachable
 * while the model is unusable (missing weights, say), and collapsing those into
 * one light would hide exactly the distinction a user needs to act on.
 */
export function TopBar({
  health,
  checking,
  error,
  onRefresh,
  onOpenSettings,
}: {
  health: Health | null
  checking: boolean
  error: string | null
  onRefresh: () => void
  onOpenSettings: () => void
}) {
  const connected = health !== null && error === null
  const modelReady = health?.model_ready ?? false

  return (
    <header className="flex items-center justify-between gap-4 border-b border-edge bg-panel px-5 py-3">
      <div className="min-w-0">
        <h1 className="truncate text-sm font-semibold tracking-tight">
          VisionDetect
          <span className="ml-2 font-normal text-ink-faint">
            Computer vision platform
          </span>
        </h1>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <StatusPill
          tone={connected ? (modelReady ? 'good' : 'warn') : 'bad'}
          label={
            checking
              ? 'Checking…'
              : connected
                ? modelReady
                  ? `${health?.default_model}`
                  : 'Model unavailable'
                : 'Backend offline'
          }
          title={
            health?.model_ready
              ? `${health.default_model} ready`
              : (health?.model_status ?? error ?? 'Not connected')
          }
          busy={checking}
        />

        {health?.model_ready && (
          <span className="hidden items-center gap-1.5 rounded-full border border-edge px-2.5 py-1 text-[11px] font-medium text-ink-soft sm:inline-flex">
            <Cpu size={12} />
            {health.device.toUpperCase()}
          </span>
        )}

        <button
          type="button"
          onClick={onRefresh}
          title="Re-check the backend"
          aria-label="Re-check the backend"
          className="rounded-lg p-1.5 text-ink-soft transition hover:bg-canvas hover:text-ink"
        >
          <RefreshCw size={15} className={checking ? 'animate-spin' : undefined} />
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
          className="rounded-lg p-1.5 text-ink-soft transition hover:bg-canvas hover:text-ink"
        >
          <SettingsIcon size={15} />
        </button>
      </div>
    </header>
  )
}

function StatusPill({
  tone,
  label,
  title,
  busy,
}: {
  tone: 'good' | 'warn' | 'bad'
  label: string
  title: string
  busy: boolean
}) {
  const dot = {
    good: 'bg-good',
    warn: 'bg-warn',
    bad: 'bg-bad',
  }[tone]

  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 rounded-full border border-edge px-2.5 py-1 text-[11px] font-medium text-ink-soft"
    >
      {busy ? (
        <Spinner className="h-2.5 w-2.5 text-ink-faint" />
      ) : (
        <span className={classNames('h-1.5 w-1.5 rounded-full', dot)} />
      )}
      {label}
    </span>
  )
}
