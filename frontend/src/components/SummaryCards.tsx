import {
  Boxes,
  Gauge,
  Layers,
  Maximize,
  Timer,
  TrendingUp,
} from 'lucide-react'
import type { ReactNode } from 'react'

import { milliseconds, percent } from '../lib/format'
import type { Detection, ImageMeta } from '../types'

/**
 * Headline metrics.
 *
 * Every figure is computed here from the detections currently visible, so that
 * filtering a class out of the view changes the totals rather than leaving the
 * cards describing a different set of objects than the ones on screen.
 */
export function SummaryCards({
  detections,
  image,
  processingMs,
  inferenceMs,
  device,
}: {
  detections: Detection[]
  image: ImageMeta
  processingMs: number
  inferenceMs: number
  device: string
}) {
  const confidences = detections.map((detection) => detection.confidence)
  const uniqueClasses = new Set(detections.map((d) => d.class_name)).size

  const highest = confidences.length ? Math.max(...confidences) : 0
  const average = confidences.length
    ? confidences.reduce((total, value) => total + value, 0) / confidences.length
    : 0

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
      <Card
        icon={<Boxes size={14} />}
        label="Total objects"
        value={String(detections.length)}
      />
      <Card
        icon={<Layers size={14} />}
        label="Unique classes"
        value={String(uniqueClasses)}
      />
      <Card
        icon={<TrendingUp size={14} />}
        label="Highest confidence"
        value={detections.length ? percent(highest) : '—'}
      />
      <Card
        icon={<Gauge size={14} />}
        label="Average confidence"
        value={detections.length ? percent(average) : '—'}
      />
      <Card
        icon={<Maximize size={14} />}
        label="Resolution"
        value={`${image.width} × ${image.height}`}
        hint={`${((image.width * image.height) / 1e6).toFixed(1)} MP`}
      />
      <Card
        icon={<Timer size={14} />}
        label="Processing time"
        value={milliseconds(processingMs)}
        hint={`${milliseconds(inferenceMs)} inference · ${device.toUpperCase()}`}
      />
    </div>
  )
}

function Card({
  icon,
  label,
  value,
  hint,
}: {
  icon: ReactNode
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="rounded-xl border border-edge bg-panel p-3.5">
      <div className="mb-2 flex items-center gap-1.5 text-ink-faint">
        {icon}
        <span className="text-[11px] font-medium">{label}</span>
      </div>
      <p className="truncate font-mono text-lg font-semibold tabular-nums tracking-tight text-ink">
        {value}
      </p>
      {hint && <p className="mt-0.5 truncate text-[11px] text-ink-faint">{hint}</p>}
    </div>
  )
}
