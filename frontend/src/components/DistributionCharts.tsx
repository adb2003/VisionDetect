import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { percent } from '../lib/format'
import type { Detection } from '../types'
import { Panel, PanelHeading } from './ui'

/**
 * Object distribution: counts per class, and each class's share of the total.
 *
 * Both charts are driven by the detections currently visible, so hiding a class
 * removes it from the charts as well as from the image.
 *
 * Colour here is the same per-class colour used for the bounding boxes, which
 * is what lets someone glance between the chart and the photograph and match
 * them up. Because a model's 80 classes cannot all be given distinguishable
 * hues, colour is never the only label: every bar is named on the axis and
 * every segment is named in the legend.
 */

const MAX_SLICES = 8

interface Slice {
  name: string
  count: number
  colour: string
  share: number
}

function buildSlices(detections: Detection[]): Slice[] {
  const byClass = new Map<string, { count: number; colour: string }>()
  for (const detection of detections) {
    const existing = byClass.get(detection.class_name)
    if (existing) {
      existing.count += 1
    } else {
      byClass.set(detection.class_name, { count: 1, colour: detection.color })
    }
  }

  const total = detections.length || 1
  const ordered = [...byClass.entries()]
    .map(([name, value]) => ({
      name,
      count: value.count,
      colour: value.colour,
      share: value.count / total,
    }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))

  if (ordered.length <= MAX_SLICES) return ordered

  // Beyond a handful of categories the colours stop being tellable apart and
  // the chart stops being readable, so the tail is folded into one grey slice
  // rather than generating more hues. The table below still lists every class.
  const head = ordered.slice(0, MAX_SLICES - 1)
  const tail = ordered.slice(MAX_SLICES - 1)
  const tailCount = tail.reduce((sum, item) => sum + item.count, 0)

  return [
    ...head,
    {
      name: `Other (${tail.length})`,
      count: tailCount,
      colour: '#9ca3af',
      share: tailCount / total,
    },
  ]
}

export function DistributionCharts({ detections }: { detections: Detection[] }) {
  const slices = useMemo(() => buildSlices(detections), [detections])

  if (detections.length === 0) {
    return (
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <PanelHeading title="Objects per class" />
          <NoData />
        </Panel>
        <Panel>
          <PanelHeading title="Class proportions" />
          <NoData />
        </Panel>
      </div>
    )
  }

  // Horizontal bars: class names read left-to-right at full size instead of
  // being rotated or truncated under a vertical axis.
  const chartHeight = Math.max(160, slices.length * 34 + 24)

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel>
        <PanelHeading
          title="Objects per class"
          subtitle={`${detections.length} ${detections.length === 1 ? 'object' : 'objects'} across ${slices.length} ${slices.length === 1 ? 'group' : 'groups'}`}
        />
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart
            data={slices}
            layout="vertical"
            margin={{ top: 0, right: 28, bottom: 0, left: 0 }}
            barCategoryGap={6}
          >
            <XAxis
              type="number"
              allowDecimals={false}
              tick={{ fontSize: 11, fill: '#6b7280' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={100}
              tick={{ fontSize: 11, fill: '#374151' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: 'rgba(0,0,0,0.04)' }}
              content={<BarTooltip />}
            />
            <Bar dataKey="count" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {slices.map((slice) => (
                <Cell key={slice.name} fill={slice.colour} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel>
        <PanelHeading
          title="Class proportions"
          subtitle="Share of all detected objects"
        />
        <div className="flex flex-col items-center gap-4 sm:flex-row">
          <ResponsiveContainer width="100%" height={200} className="max-w-[220px]">
            <PieChart>
              <Pie
                data={slices}
                dataKey="count"
                nameKey="name"
                innerRadius={52}
                outerRadius={82}
                paddingAngle={2}
                stroke="#ffffff"
                strokeWidth={2}
                isAnimationActive={false}
              >
                {slices.map((slice) => (
                  <Cell key={slice.name} fill={slice.colour} />
                ))}
              </Pie>
              <Tooltip content={<BarTooltip />} />
            </PieChart>
          </ResponsiveContainer>

          {/* A real legend, not a colour key: names and numbers carry the
              meaning even if the hues are hard to tell apart. */}
          <ul className="w-full flex-1 space-y-1">
            {slices.map((slice) => (
              <li
                key={slice.name}
                className="flex items-center gap-2 text-xs text-ink-soft"
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: slice.colour }}
                  aria-hidden="true"
                />
                <span className="flex-1 truncate text-ink">{slice.name}</span>
                <span className="font-mono tabular-nums">{slice.count}</span>
                <span className="w-11 text-right font-mono tabular-nums text-ink-faint">
                  {percent(slice.share, 0)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </Panel>
    </div>
  )
}

function BarTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: { payload: Slice }[]
}) {
  if (!active || !payload?.length) return null
  const slice = payload[0].payload

  return (
    <div className="rounded-lg border border-edge bg-panel px-2.5 py-1.5 shadow-lg">
      <p className="text-xs font-medium text-ink">{slice.name}</p>
      <p className="mt-0.5 font-mono text-[11px] tabular-nums text-ink-soft">
        {slice.count} {slice.count === 1 ? 'object' : 'objects'} ·{' '}
        {percent(slice.share, 0)}
      </p>
    </div>
  )
}

function NoData() {
  return (
    <div className="grid h-36 place-items-center rounded-lg border border-dashed border-edge">
      <p className="text-xs text-ink-faint">No objects to chart</p>
    </div>
  )
}
