/** Shared presentational primitives, so spacing and borders stay consistent. */

import type { ReactNode } from 'react'

import { classNames } from '../lib/format'

export function Panel({
  children,
  className,
  padded = true,
}: {
  children: ReactNode
  className?: string
  padded?: boolean
}) {
  return (
    <section
      className={classNames(
        'rounded-xl border border-edge bg-panel shadow-[0_1px_2px_rgba(16,24,40,0.04)]',
        padded && 'p-5',
        className,
      )}
    >
      {children}
    </section>
  )
}

export function PanelHeading({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: string
  actions?: ReactNode
}) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-ink-soft">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary:
    'bg-accent text-white hover:brightness-110 disabled:bg-edge-strong disabled:text-ink-faint',
  secondary:
    'border border-edge bg-panel text-ink hover:bg-canvas disabled:text-ink-faint',
  ghost: 'text-ink-soft hover:bg-canvas hover:text-ink',
  danger: 'border border-edge bg-panel text-bad hover:bg-red-50',
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  title,
  type = 'button',
  className,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: ButtonVariant
  disabled?: boolean
  title?: string
  type?: 'button' | 'submit'
  className?: string
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={classNames(
        'inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5',
        'text-xs font-medium transition disabled:cursor-not-allowed',
        BUTTON_STYLES[variant],
        className,
      )}
    >
      {children}
    </button>
  )
}

/** A segmented control; used for the view mode switch. */
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: string; icon?: ReactNode }[]
  onChange: (value: T) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-edge bg-canvas p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
          className={classNames(
            'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition',
            value === option.value
              ? 'bg-panel text-ink shadow-[0_1px_2px_rgba(16,24,40,0.08)]'
              : 'text-ink-soft hover:text-ink',
          )}
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-1.5">
      <span className="text-xs text-ink-soft">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={classNames(
          'relative h-5 w-9 shrink-0 rounded-full transition',
          checked ? 'bg-accent' : 'bg-edge-strong',
        )}
      >
        <span
          className={classNames(
            'absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all',
            checked ? 'left-4.5' : 'left-0.5',
          )}
        />
      </button>
    </label>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  hint?: string
}) {
  return (
    <div className="py-2">
      <div className="mb-1.5 flex items-baseline justify-between">
        <label className="text-xs font-medium text-ink">{label}</label>
        <span className="font-mono text-xs tabular-nums text-ink-soft">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint && <p className="mt-1 text-[11px] leading-snug text-ink-faint">{hint}</p>}
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={classNames(
        'inline-block animate-spin rounded-full border-2 border-current border-t-transparent',
        className ?? 'h-3.5 w-3.5',
      )}
      aria-hidden="true"
    />
  )
}

export function EmptyState({
  icon,
  title,
  description,
}: {
  icon: ReactNode
  title: string
  description: string
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      <div className="mb-3 rounded-full bg-canvas p-3 text-ink-faint">{icon}</div>
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="mt-1 max-w-sm text-xs leading-relaxed text-ink-soft">{description}</p>
    </div>
  )
}

export function ErrorNotice({
  message,
  onDismiss,
}: {
  message: string
  onDismiss?: () => void
}) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3"
    >
      <p className="flex-1 text-xs leading-relaxed text-red-900">{message}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="text-xs font-medium text-red-700 hover:text-red-900"
        >
          Dismiss
        </button>
      )}
    </div>
  )
}
