import { Eye, EyeOff } from 'lucide-react'

import { classNames, percent } from '../lib/format'
import { contrastText } from '../lib/colour'
import type { ClassSummary } from '../types'
import { Button, Panel, PanelHeading } from './ui'

/**
 * Per-class visibility toggles.
 *
 * Hiding a class removes it from the overlay, the summary cards, the charts and
 * the table at once, so every number on screen always describes exactly the set
 * of objects being shown.
 */
export function ClassFilter({
  classes,
  hidden,
  onToggle,
  onSetAll,
}: {
  classes: ClassSummary[]
  hidden: Set<string>
  onToggle: (className: string) => void
  onSetAll: (allVisible: boolean) => void
}) {
  if (classes.length === 0) return null

  const visibleCount = classes.length - hidden.size

  return (
    <Panel>
      <PanelHeading
        title="Object categories"
        subtitle={`${visibleCount} of ${classes.length} shown`}
        actions={
          <Button
            variant="ghost"
            onClick={() => onSetAll(hidden.size > 0)}
            title={hidden.size > 0 ? 'Show every class' : 'Hide every class'}
          >
            {hidden.size > 0 ? <Eye size={13} /> : <EyeOff size={13} />}
            {hidden.size > 0 ? 'Show all' : 'Hide all'}
          </Button>
        }
      />

      <ul className="space-y-1">
        {classes.map((entry) => {
          const isHidden = hidden.has(entry.class_name)
          return (
            <li key={entry.class_name}>
              <button
                type="button"
                onClick={() => onToggle(entry.class_name)}
                aria-pressed={!isHidden}
                className={classNames(
                  'flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition',
                  isHidden ? 'opacity-45 hover:opacity-70' : 'hover:bg-canvas',
                )}
              >
                <span
                  className="grid h-5 w-5 shrink-0 place-items-center rounded font-mono text-[10px] font-semibold"
                  style={{
                    backgroundColor: entry.color,
                    color: contrastText(entry.color),
                  }}
                >
                  {entry.count}
                </span>
                <span className="flex-1 truncate text-xs font-medium text-ink">
                  {entry.class_name}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-ink-faint">
                  {percent(entry.avg_confidence, 0)}
                </span>
                {isHidden ? (
                  <EyeOff size={12} className="text-ink-faint" />
                ) : (
                  <Eye size={12} className="text-ink-faint" />
                )}
              </button>
            </li>
          )
        })}
      </ul>

      <p className="mt-2 border-t border-edge pt-2 text-[11px] leading-relaxed text-ink-faint">
        Percentages are the mean confidence for that class.
      </p>
    </Panel>
  )
}
