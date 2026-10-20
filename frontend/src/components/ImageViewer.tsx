import { useEffect, useRef, useState } from 'react'
import { Columns2, Image as ImageIcon, Maximize2, Minus, Plus, SquareDashed } from 'lucide-react'

import { contrastText } from '../lib/colour'
import { classNames, percent } from '../lib/format'
import type { Detection, ImageMeta, ViewMode, ViewOptions } from '../types'
import { Button, SegmentedControl } from './ui'

const ZOOM_STEPS = [1, 1.5, 2, 3, 4]

/**
 * The image with its detections drawn over it.
 *
 * The overlay is rendered in the browser rather than shown as the server's
 * annotated PNG, for one reason that matters: toggling a class or a label has
 * to be instant, and a round trip per checkbox is not. Boxes are an SVG whose
 * `viewBox` is the image's own pixel space, so a coordinate from the API is
 * placed without any arithmetic here -- there is no second chance to get the
 * scaling wrong. Labels are HTML positioned in percentages so their text stays
 * the same size at every zoom level.
 *
 * The server-rendered PNG is still what gets downloaded, and it is drawn from
 * the same coordinates and the same per-class colours.
 */
export function ImageViewer({
  imageUrl,
  image,
  detections,
  view,
  mode,
  onModeChange,
  selectedId,
  onSelect,
}: {
  imageUrl: string
  image: ImageMeta
  detections: Detection[]
  view: ViewOptions
  mode: ViewMode
  onModeChange: (mode: ViewMode) => void
  selectedId: number | null
  onSelect: (objectId: number | null) => void
}) {
  const [zoom, setZoom] = useState(1)
  const scrollRef = useRef<HTMLDivElement>(null)

  // A new image should not inherit the previous one's magnification.
  useEffect(() => {
    setZoom(1)
  }, [imageUrl])

  const zoomIndex = ZOOM_STEPS.indexOf(zoom)
  const canZoomIn = zoomIndex < ZOOM_STEPS.length - 1
  const canZoomOut = zoomIndex > 0

  return (
    <div className="flex h-full flex-col">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl<ViewMode>
          value={mode}
          onChange={onModeChange}
          options={[
            { value: 'annotated', label: 'Annotated', icon: <SquareDashed size={13} /> },
            { value: 'original', label: 'Original', icon: <ImageIcon size={13} /> },
            { value: 'split', label: 'Side by side', icon: <Columns2 size={13} /> },
          ]}
        />

        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[11px] tabular-nums text-ink-faint">
            {image.width} × {image.height}
          </span>
          <div className="flex items-center rounded-lg border border-edge">
            <button
              type="button"
              aria-label="Zoom out"
              disabled={!canZoomOut}
              onClick={() => setZoom(ZOOM_STEPS[zoomIndex - 1])}
              className="px-2 py-1.5 text-ink-soft transition hover:text-ink disabled:text-edge-strong"
            >
              <Minus size={13} />
            </button>
            <span className="w-10 text-center font-mono text-[11px] tabular-nums text-ink-soft">
              {zoom}×
            </span>
            <button
              type="button"
              aria-label="Zoom in"
              disabled={!canZoomIn}
              onClick={() => setZoom(ZOOM_STEPS[zoomIndex + 1])}
              className="px-2 py-1.5 text-ink-soft transition hover:text-ink disabled:text-edge-strong"
            >
              <Plus size={13} />
            </button>
          </div>
          {zoom !== 1 && (
            <Button variant="ghost" onClick={() => setZoom(1)} title="Fit to width">
              <Maximize2 size={13} />
            </Button>
          )}
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-auto rounded-xl border border-edge bg-[repeating-conic-gradient(#f4f4f5_0_25%,#fafafa_0_50%)] bg-[length:16px_16px] p-2"
      >
        {mode === 'split' ? (
          <div className="flex gap-2" style={{ width: `${zoom * 100}%` }}>
            <Pane label="Original" className="flex-1">
              <img
                src={imageUrl}
                alt={`Original upload ${image.filename}`}
                className="block w-full"
              />
            </Pane>
            <Pane label={`Annotated · ${detections.length} shown`} className="flex-1">
              <Canvas
                imageUrl={imageUrl}
                image={image}
                detections={detections}
                view={view}
                selectedId={selectedId}
                onSelect={onSelect}
                compact
              />
            </Pane>
          </div>
        ) : (
          <div style={{ width: `${zoom * 100}%` }}>
            {mode === 'original' ? (
              <img
                src={imageUrl}
                alt={`Original upload ${image.filename}`}
                className="block w-full rounded-lg"
              />
            ) : (
              <Canvas
                imageUrl={imageUrl}
                image={image}
                detections={detections}
                view={view}
                selectedId={selectedId}
                onSelect={onSelect}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Pane({
  label,
  children,
  className,
}: {
  label: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={classNames('min-w-0', className)}>
      <p className="mb-1.5 text-[11px] font-medium text-ink-soft">{label}</p>
      {children}
    </div>
  )
}

function Canvas({
  imageUrl,
  image,
  detections,
  view,
  selectedId,
  onSelect,
  compact = false,
}: {
  imageUrl: string
  image: ImageMeta
  detections: Detection[]
  view: ViewOptions
  selectedId: number | null
  onSelect: (objectId: number | null) => void
  compact?: boolean
}) {
  const hasSelection = selectedId !== null

  return (
    <div className="relative block overflow-hidden rounded-lg">
      <img
        src={imageUrl}
        alt={`${image.filename} with ${detections.length} detected objects`}
        className="block w-full"
        onClick={() => onSelect(null)}
      />

      {view.show_boxes && (
        <svg
          viewBox={`0 0 ${image.width} ${image.height}`}
          className="pointer-events-none absolute inset-0 h-full w-full"
          aria-hidden="true"
        >
          {detections.map((detection) => {
            const selected = detection.object_id === selectedId
            const { x_min, y_min } = detection.bbox
            return (
              <rect
                key={detection.object_id}
                x={x_min}
                y={y_min}
                width={detection.width}
                height={detection.height}
                fill={selected ? detection.color : 'none'}
                fillOpacity={selected ? 0.16 : 0}
                stroke={detection.color}
                strokeWidth={selected ? 3 : 2}
                strokeOpacity={hasSelection && !selected ? 0.35 : 1}
                vectorEffect="non-scaling-stroke"
                rx={1}
              />
            )
          })}
        </svg>
      )}

      {view.show_labels &&
        detections.map((detection) => {
          const selected = detection.object_id === selectedId
          const { x_min, y_min } = detection.bbox
          // Percentages rather than pixels: the label tracks the box at any
          // zoom while its own text stays a constant size on screen.
          const left = (x_min / image.width) * 100
          const top = (y_min / image.height) * 100
          const nearTop = top < 6

          return (
            <button
              key={detection.object_id}
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                onSelect(selected ? null : detection.object_id)
              }}
              style={{
                left: `${left}%`,
                top: `${top}%`,
                backgroundColor: detection.color,
                color: contrastText(detection.color),
                transform: nearTop ? 'translateY(0)' : 'translateY(-100%)',
                opacity: hasSelection && !selected ? 0.45 : 1,
              }}
              className={classNames(
                'absolute max-w-[60%] truncate rounded-sm px-1 py-px text-left font-medium',
                'shadow-[0_1px_2px_rgba(0,0,0,0.35)] transition-opacity',
                compact ? 'text-[9px]' : 'text-[10px]',
                selected && 'ring-2 ring-white',
              )}
              title={`${detection.class_name} · ${percent(detection.confidence)}`}
            >
              {detection.class_name}
              {view.show_confidence && (
                <span className="ml-1 font-mono tabular-nums opacity-90">
                  {Math.round(detection.confidence * 100)}%
                </span>
              )}
            </button>
          )
        })}
    </div>
  )
}
