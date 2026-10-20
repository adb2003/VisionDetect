import { useEffect, useState } from 'react'
import { Activity, Boxes, Clock, Cpu, ScanSearch, Server } from 'lucide-react'
import type { ReactNode } from 'react'

import { Button, EmptyState, Panel, PanelHeading } from '../components/ui'
import { milliseconds, timestamp } from '../lib/format'
import { getHistory } from '../services/api'
import type { DetectionResult, Health, HistoryEntry } from '../types'

/**
 * Overview of the service and of recent work.
 *
 * Shows what is true right now -- the backend's own health report and the
 * recorded history -- rather than a mock of what a dashboard might contain.
 */
export function DashboardPage({
  health,
  result,
  onStart,
}: {
  health: Health | null
  result: DetectionResult | null
  onStart: () => void
}) {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [total, setTotal] = useState(0)

  useEffect(() => {
    let cancelled = false
    getHistory(5)
      .then((response) => {
        if (cancelled) return
        setEntries(response.entries)
        setTotal(response.total)
      })
      .catch(() => {
        /* The dashboard degrades to the live panels if history is unreachable. */
      })
    return () => {
      cancelled = true
    }
  }, [result])

  const totalObjects = entries.reduce((sum, entry) => sum + entry.total_objects, 0)
  const averageMs = entries.length
    ? entries.reduce((sum, entry) => sum + entry.processing_time_ms, 0) / entries.length
    : 0

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">Dashboard</h2>
          <p className="mt-0.5 text-xs text-ink-soft">
            Service status and recent detection activity.
          </p>
        </div>
        <Button variant="primary" onClick={onStart}>
          <ScanSearch size={14} />
          New detection
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          icon={<Server size={14} />}
          label="Backend"
          value={health ? (health.status === 'ok' ? 'Online' : 'Degraded') : 'Offline'}
          hint={health ? `v${health.version}` : 'Not reachable'}
        />
        <Stat
          icon={<Cpu size={14} />}
          label="Inference device"
          value={health?.device.toUpperCase() ?? '—'}
          hint={health?.default_model ?? 'No model'}
        />
        <Stat
          icon={<Activity size={14} />}
          label="Analyses recorded"
          value={String(total)}
          hint={total ? `${totalObjects} objects in last ${entries.length}` : 'None yet'}
        />
        <Stat
          icon={<Clock size={14} />}
          label="Average time"
          value={averageMs ? milliseconds(averageMs) : '—'}
          hint="Across recent runs"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel>
          <PanelHeading
            title="Model"
            subtitle="The detector that serves requests by default"
          />
          {health?.model_ready ? (
            <dl className="space-y-2 text-xs">
              <Row label="Name" value={health.default_model} />
              <Row label="Device" value={health.device.toUpperCase()} />
              <Row
                label="Runtime providers"
                value={health.providers.join(', ') || 'none reported'}
              />
              <Row
                label="Loaded in memory"
                value={health.loaded_models.join(', ') || 'none yet'}
              />
              <Row
                label="Default thresholds"
                value={`confidence ${health.defaults.confidence_threshold}, IoU ${health.defaults.iou_threshold}`}
              />
            </dl>
          ) : (
            <p className="rounded-lg bg-canvas px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
              {health?.model_status ??
                'The backend is not reachable. Start it on port 8000 and refresh.'}
            </p>
          )}
        </Panel>

        <Panel padded={false}>
          <div className="p-5 pb-3">
            <PanelHeading title="Recent analyses" subtitle="Metadata only" />
          </div>
          {entries.length ? (
            <ul className="divide-y divide-edge border-t border-edge">
              {entries.map((entry) => (
                <li key={entry.analysis_id} className="flex items-center gap-3 px-5 py-2.5">
                  <Boxes size={14} className="shrink-0 text-ink-faint" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-ink">
                      {entry.filename}
                    </p>
                    <p className="truncate text-[11px] text-ink-faint">
                      {entry.top_classes || 'no objects'} · {timestamp(entry.timestamp)}
                    </p>
                  </div>
                  <span className="font-mono text-xs tabular-nums text-ink-soft">
                    {entry.total_objects}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={<Clock size={20} />}
              title="No analyses yet"
              description="Runs you complete will be listed here with their object counts and timings."
            />
          )}
        </Panel>
      </div>
    </div>
  )
}

function Stat({
  icon,
  label,
  value,
  hint,
}: {
  icon: ReactNode
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="rounded-xl border border-edge bg-panel p-3.5">
      <div className="mb-2 flex items-center gap-1.5 text-ink-faint">
        {icon}
        <span className="text-[11px] font-medium">{label}</span>
      </div>
      <p className="truncate text-base font-semibold tracking-tight text-ink">{value}</p>
      <p className="mt-0.5 truncate text-[11px] text-ink-faint">{hint}</p>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-edge pb-2 last:border-0">
      <dt className="shrink-0 text-ink-faint">{label}</dt>
      <dd className="truncate text-right font-medium text-ink">{value}</dd>
    </div>
  )
}
