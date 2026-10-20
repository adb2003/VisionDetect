import { useCallback, useEffect, useState } from 'react'
import { Clock, RefreshCw, Trash2 } from 'lucide-react'

import { Button, EmptyState, ErrorNotice, Panel, PanelHeading } from '../components/ui'
import { milliseconds, timestamp } from '../lib/format'
import { clearHistory, deleteHistoryEntry, getHistory } from '../services/api'
import type { HistoryEntry } from '../types'

/**
 * Past analyses.
 *
 * Metadata only -- the images themselves are not retained. The "result" column
 * says whether the full result is still in the server's memory cache, which is
 * the honest distinction between a record that happened and one you can still
 * open.
 */
export function HistoryPage({ refreshToken }: { refreshToken: number }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const response = await getHistory(200)
      setEntries(response.entries)
      setTotal(response.total)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not load history.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshToken])

  async function remove(analysisId: string) {
    await deleteHistoryEntry(analysisId)
    await load()
  }

  async function removeAll() {
    await clearHistory()
    await load()
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">
            Detection History
          </h2>
          <p className="mt-0.5 text-xs text-ink-soft">
            {total} {total === 1 ? 'analysis' : 'analyses'} recorded. Uploaded
            images are not stored.
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => void load()} disabled={loading}>
            <RefreshCw size={13} className={loading ? 'animate-spin' : undefined} />
            Refresh
          </Button>
          {entries.length > 0 && (
            <Button variant="danger" onClick={() => void removeAll()}>
              <Trash2 size={13} />
              Clear all
            </Button>
          )}
        </div>
      </div>

      {error && <ErrorNotice message={error} onDismiss={() => setError(null)} />}

      <Panel padded={false}>
        <div className="p-5 pb-3">
          <PanelHeading title="Analyses" subtitle="Most recent first" />
        </div>

        {entries.length === 0 && !loading ? (
          <EmptyState
            icon={<Clock size={20} />}
            title="Nothing recorded yet"
            description="Each detection you run is logged here with its filename, object counts and processing time."
          />
        ) : (
          <div className="overflow-x-auto border-t border-edge">
            <table className="w-full border-collapse text-xs">
              <thead className="bg-canvas">
                <tr>
                  {[
                    'Filename',
                    'When',
                    'Model',
                    'Objects',
                    'Classes',
                    'Resolution',
                    'Time',
                    '',
                  ].map((heading) => (
                    <th
                      key={heading}
                      scope="col"
                      className="border-b border-edge px-3 py-2 text-left font-medium text-ink-soft"
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={entry.analysis_id} className="border-b border-edge/60">
                    <td className="max-w-[14rem] px-3 py-2">
                      <p className="truncate font-medium text-ink">{entry.filename}</p>
                      <p className="truncate text-[11px] text-ink-faint">
                        {entry.top_classes || 'no objects detected'}
                      </p>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-ink-soft">
                      {timestamp(entry.timestamp)}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-ink-soft">
                      {entry.model}
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums text-ink">
                      {entry.total_objects}
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums text-ink-soft">
                      {entry.unique_classes}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] tabular-nums whitespace-nowrap text-ink-soft">
                      {entry.width} × {entry.height}
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums whitespace-nowrap text-ink-soft">
                      {milliseconds(entry.processing_time_ms)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button
                        type="button"
                        onClick={() => void remove(entry.analysis_id)}
                        aria-label={`Delete history entry for ${entry.filename}`}
                        className="rounded p-1 text-ink-faint transition hover:bg-red-50 hover:text-bad"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
