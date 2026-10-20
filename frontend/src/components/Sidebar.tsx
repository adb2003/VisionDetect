import { Boxes, Clock, LayoutDashboard, ScanSearch, Settings } from 'lucide-react'

import { classNames } from '../lib/format'

export type Page = 'dashboard' | 'detect' | 'history' | 'settings'

const NAV: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'detect', label: 'Object Detection', icon: ScanSearch },
  { id: 'history', label: 'Detection History', icon: Clock },
  { id: 'settings', label: 'Settings', icon: Settings },
]

export function Sidebar({
  page,
  onNavigate,
  detectionCount,
}: {
  page: Page
  onNavigate: (page: Page) => void
  detectionCount: number | null
}) {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-edge bg-panel max-md:hidden">
      <div className="flex items-center gap-2.5 border-b border-edge px-5 py-4">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-white">
          <Boxes size={17} strokeWidth={2.2} />
        </span>
        <div className="leading-tight">
          <p className="text-sm font-semibold tracking-tight">VisionDetect</p>
          <p className="text-[11px] text-ink-faint">Object detection</p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        {NAV.map((item) => {
          const Icon = item.icon
          const active = page === item.id
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              aria-current={active ? 'page' : undefined}
              className={classNames(
                'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs font-medium transition',
                active
                  ? 'bg-accent-soft text-accent'
                  : 'text-ink-soft hover:bg-canvas hover:text-ink',
              )}
            >
              <Icon size={15} strokeWidth={2} />
              <span className="flex-1">{item.label}</span>
              {item.id === 'detect' && detectionCount !== null && (
                <span className="rounded-full bg-panel px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-ink-soft ring-1 ring-edge">
                  {detectionCount}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      <div className="border-t border-edge p-4">
        <p className="text-[11px] leading-relaxed text-ink-faint">
          Images are processed in memory and are not stored on the server by
          default.
        </p>
      </div>
    </aside>
  )
}

/** Compact nav for narrow screens, where the sidebar is hidden. */
export function MobileNav({
  page,
  onNavigate,
}: {
  page: Page
  onNavigate: (page: Page) => void
}) {
  return (
    <nav className="flex gap-1 overflow-x-auto border-b border-edge bg-panel px-3 py-2 md:hidden">
      {NAV.map((item) => {
        const Icon = item.icon
        const active = page === item.id
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onNavigate(item.id)}
            className={classNames(
              'flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium',
              active ? 'bg-accent-soft text-accent' : 'text-ink-soft',
            )}
          >
            <Icon size={14} />
            {item.label}
          </button>
        )
      })}
    </nav>
  )
}
