import { ChevronDown, Cpu, Play, RotateCcw } from 'lucide-react'

import { classNames } from '../lib/format'
import type {
  DetectionControls,
  DetectionResult,
  ModelsResponse,
  ViewOptions,
} from '../types'
import { Button, Slider, Spinner, Toggle } from './ui'

/**
 * Detection settings.
 *
 * The distinction the panel is built around: the two thresholds and the model
 * change what the network does and need a re-run, while the label switches only
 * change what is drawn and take effect immediately. Mixing those would either
 * make every checkbox cost a round trip or make the thresholds look like they
 * applied when they had not.
 */
export function SettingsPanel({
  open,
  onToggle,
  controls,
  onControlsChange,
  view,
  onViewChange,
  models,
  result,
  busy,
  onApply,
  onReset,
  dirty,
}: {
  open: boolean
  onToggle: () => void
  controls: DetectionControls
  onControlsChange: (controls: DetectionControls) => void
  view: ViewOptions
  onViewChange: (view: ViewOptions) => void
  models: ModelsResponse | null
  result: DetectionResult | null
  busy: boolean
  onApply: () => void
  onReset: () => void
  dirty: boolean
}) {
  const selected = models?.models.find((model) => model.name === controls.model)
  const ranAt = result?.settings.confidence_threshold

  return (
    <div className="rounded-xl border border-edge bg-panel">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-5 py-3.5 text-left"
      >
        <span className="text-sm font-semibold tracking-tight text-ink">
          Detection settings
        </span>
        <span className="flex items-center gap-2">
          {dirty && (
            <span className="rounded-full bg-warn/15 px-2 py-0.5 text-[10px] font-medium text-warn">
              Not applied
            </span>
          )}
          <ChevronDown
            size={15}
            className={classNames(
              'text-ink-faint transition-transform',
              open && 'rotate-180',
            )}
          />
        </span>
      </button>

      {open && (
        <div className="space-y-4 border-t border-edge px-5 py-4">
          <div>
            <label
              htmlFor="model-select"
              className="mb-1.5 block text-xs font-medium text-ink"
            >
              Model
            </label>
            <select
              id="model-select"
              value={controls.model}
              onChange={(event) =>
                onControlsChange({ ...controls, model: event.target.value })
              }
              className="w-full rounded-lg border border-edge bg-panel px-2.5 py-2 text-xs text-ink outline-none focus:border-accent"
            >
              {(models?.models ?? []).map((model) => (
                <option
                  key={model.name}
                  value={model.name}
                  disabled={model.status === 'missing' || model.status === 'error'}
                >
                  {model.label}
                  {model.status === 'missing' ? ' — weights missing' : ''}
                  {model.status === 'error' ? ' — failed to load' : ''}
                </option>
              ))}
            </select>

            {selected && (
              <div className="mt-2 space-y-1 rounded-lg bg-canvas px-2.5 py-2">
                <div className="flex items-center justify-between gap-2">
                  <StatusDot status={selected.status} />
                  {result && (
                    <span className="inline-flex items-center gap-1 text-[11px] text-ink-soft">
                      <Cpu size={11} />
                      {result.model_info.device.toUpperCase()}
                    </span>
                  )}
                </div>
                <p className="text-[11px] leading-relaxed text-ink-faint">
                  {selected.detail}
                </p>
                {selected.class_count !== null && (
                  <p className="text-[11px] text-ink-faint">
                    {selected.class_count} detectable classes · {selected.backend}
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="border-t border-edge pt-1">
            <Slider
              label="Confidence threshold"
              value={controls.confidence_threshold}
              min={0.05}
              max={0.95}
              step={0.05}
              onChange={(value) =>
                onControlsChange({ ...controls, confidence_threshold: value })
              }
              hint={
                ranAt !== undefined && controls.confidence_threshold < ranAt
                  ? `Below the ${ranAt.toFixed(2)} this image was run at — apply to find weaker detections.`
                  : 'Minimum score an object must reach to be reported.'
              }
            />
            <Slider
              label="IoU threshold"
              value={controls.iou_threshold}
              min={0.05}
              max={0.95}
              step={0.05}
              onChange={(value) =>
                onControlsChange({ ...controls, iou_threshold: value })
              }
              hint="How much two boxes may overlap before one is suppressed."
            />
          </div>

          <div className="border-t border-edge pt-2">
            <p className="mb-1 text-xs font-medium text-ink">Display</p>
            <Toggle
              label="Show bounding boxes"
              checked={view.show_boxes}
              onChange={(value) => onViewChange({ ...view, show_boxes: value })}
            />
            <Toggle
              label="Show labels"
              checked={view.show_labels}
              onChange={(value) => onViewChange({ ...view, show_labels: value })}
            />
            <Toggle
              label="Show confidence scores"
              checked={view.show_confidence}
              onChange={(value) =>
                onViewChange({ ...view, show_confidence: value })
              }
            />
          </div>

          <div className="flex gap-2 border-t border-edge pt-3">
            <Button
              variant="primary"
              onClick={onApply}
              disabled={busy || !result}
              className="flex-1 py-2"
              title={
                result
                  ? 'Re-run the model on this image with these settings'
                  : 'Run a detection first'
              }
            >
              {busy ? (
                <>
                  <Spinner className="h-3.5 w-3.5" />
                  Applying…
                </>
              ) : (
                <>
                  <Play size={13} />
                  Apply settings
                </>
              )}
            </Button>
            <Button onClick={onReset} disabled={busy} title="Restore defaults">
              <RotateCcw size={13} />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusDot({ status }: { status: string }) {
  const tone =
    status === 'loaded'
      ? 'bg-good'
      : status === 'available'
        ? 'bg-warn'
        : 'bg-bad'
  const label =
    status === 'loaded'
      ? 'Loaded'
      : status === 'available'
        ? 'Ready to load'
        : status === 'missing'
          ? 'Weights missing'
          : 'Error'

  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-ink-soft">
      <span className={classNames('h-1.5 w-1.5 rounded-full', tone)} />
      {label}
    </span>
  )
}
