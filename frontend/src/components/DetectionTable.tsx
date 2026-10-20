import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ChevronsUpDown, Search } from 'lucide-react'

import { classNames, percent } from '../lib/format'
import type { Detection } from '../types'
import { Panel, PanelHeading } from './ui'

type SortKey = 'object_id' | 'class_name' | 'confidence' | 'width' | 'height'
type Direction = 'asc' | 'desc'

const COLUMNS: { key: SortKey | null; label: string; align?: 'right' }[] = [
  { key: 'object_id', label: 'ID' },
  { key: 'class_name', label: 'Class' },
  { key: 'confidence', label: 'Confidence', align: 'right' },
  { key: null, label: 'Bounding box' },
  { key: 'width', label: 'Width', align: 'right' },
  { key: 'height', label: 'Height', align: 'right' },
]

/**
 * The per-object table.
 *
 * Doubles as the accessible view of the charts: every class name, count and
 * confidence is readable as text here, so none of the information above
 * depends on telling two colours apart.
 */
export function DetectionTable({
  detections,
  selectedId,
  onSelect,
}: {
  detections: Detection[]
  selectedId: number | null
  onSelect: (objectId: number | null) => void
}) {
  const [query, setQuery] = useState('')
  const [classFilter, setClassFilter] = useState<string>('all')
  const [sortKey, setSortKey] = useState<SortKey>('object_id')
  const [direction, setDirection] = useState<Direction>('asc')

  const classNameOptions = useMemo(
    () => [...new Set(detections.map((d) => d.class_name))].sort(),
    [detections],
  )

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const filtered = detections.filter((detection) => {
      const matchesQuery =
        !needle || detection.class_name.toLowerCase().includes(needle)
      const matchesClass =
        classFilter === 'all' || detection.class_name === classFilter
      return matchesQuery && matchesClass
    })

    const sign = direction === 'asc' ? 1 : -1
    return [...filtered].sort((a, b) => {
      const left = a[sortKey]
      const right = b[sortKey]
      if (typeof left === 'string' && typeof right === 'string') {
        return sign * left.localeCompare(right)
      }
      return sign * (Number(left) - Number(right))
    })
  }, [detections, query, classFilter, sortKey, direction])

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setDirection((current) => (current === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      // Confidence is the one column people want highest-first by default.
      setDirection(key === 'confidence' ? 'desc' : 'asc')
    }
  }

  return (
    <Panel padded={false}>
      <div className="p-5 pb-3">
        <PanelHeading
          title="Detections"
          subtitle={
            rows.length === detections.length
              ? `${detections.length} ${detections.length === 1 ? 'object' : 'objects'}`
              : `${rows.length} of ${detections.length} objects`
          }
          actions={
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search
                  size={13}
                  className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-faint"
                />
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search class…"
                  aria-label="Search by class name"
                  className="w-36 rounded-lg border border-edge bg-panel py-1.5 pr-2.5 pl-7 text-xs text-ink outline-none placeholder:text-ink-faint focus:border-accent"
                />
              </div>
              <select
                value={classFilter}
                onChange={(event) => setClassFilter(event.target.value)}
                aria-label="Filter by class"
                className="rounded-lg border border-edge bg-panel px-2 py-1.5 text-xs text-ink outline-none focus:border-accent"
              >
                <option value="all">All classes</option>
                {classNameOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          }
        />
      </div>

      <div className="max-h-[26rem] overflow-auto border-t border-edge">
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-canvas">
            <tr>
              {COLUMNS.map((column) => (
                <th
                  key={column.label}
                  scope="col"
                  className={classNames(
                    'border-b border-edge px-3 py-2 font-medium text-ink-soft',
                    column.align === 'right' ? 'text-right' : 'text-left',
                  )}
                >
                  {column.key ? (
                    <button
                      type="button"
                      onClick={() => toggleSort(column.key as SortKey)}
                      className={classNames(
                        'inline-flex items-center gap-1 transition hover:text-ink',
                        column.align === 'right' && 'flex-row-reverse',
                      )}
                    >
                      {column.label}
                      <SortIcon
                        active={sortKey === column.key}
                        direction={direction}
                      />
                    </button>
                  ) : (
                    column.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((detection) => {
              const selected = detection.object_id === selectedId
              return (
                <tr
                  key={detection.object_id}
                  onClick={() =>
                    onSelect(selected ? null : detection.object_id)
                  }
                  className={classNames(
                    'cursor-pointer border-b border-edge/60 transition',
                    selected ? 'bg-accent-soft' : 'hover:bg-canvas',
                  )}
                  title="Click to highlight this object in the image"
                >
                  <td className="px-3 py-2 font-mono tabular-nums text-ink-soft">
                    {detection.object_id}
                  </td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-sm"
                        style={{ backgroundColor: detection.color }}
                        aria-hidden="true"
                      />
                      <span className="font-medium text-ink">
                        {detection.class_name}
                      </span>
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <ConfidenceCell value={detection.confidence} />
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] tabular-nums text-ink-soft">
                    {detection.bbox.x_min}, {detection.bbox.y_min},{' '}
                    {detection.bbox.x_max}, {detection.bbox.y_max}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-ink-soft">
                    {detection.width}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-ink-soft">
                    {detection.height}
                  </td>
                </tr>
              )
            })}

            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-3 py-10 text-center text-xs text-ink-faint"
                >
                  {detections.length === 0
                    ? 'No objects were detected above the confidence threshold.'
                    : 'No detections match this search.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function ConfidenceCell({ value }: { value: number }) {
  return (
    <span className="inline-flex items-center justify-end gap-2">
      <span className="h-1 w-10 overflow-hidden rounded-full bg-edge">
        <span
          className="block h-full rounded-full bg-accent"
          style={{ width: `${Math.max(3, value * 100)}%` }}
        />
      </span>
      <span className="w-11 text-right font-mono tabular-nums text-ink">
        {percent(value)}
      </span>
    </span>
  )
}

function SortIcon({ active, direction }: { active: boolean; direction: Direction }) {
  if (!active) return <ChevronsUpDown size={11} className="text-ink-faint" />
  return direction === 'asc' ? <ArrowUp size={11} /> : <ArrowDown size={11} />
}
