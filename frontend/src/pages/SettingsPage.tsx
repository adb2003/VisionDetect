import { CheckCircle2, CircleAlert, CircleDashed, XCircle } from 'lucide-react'
import type { ReactNode } from 'react'

import { Panel, PanelHeading } from '../components/ui'
import type { Health, ModelsResponse, ModelStatus } from '../types'

/**
 * Service configuration, read from the backend.
 *
 * Deliberately read-only. These values come from environment variables on the
 * server, and a control here that appeared to change them but only affected one
 * browser session would be worse than no control at all -- so the page shows
 * what is configured and names the variable that changes it.
 */
export function SettingsPage({
  health,
  models,
}: {
  health: Health | null
  models: ModelsResponse | null
}) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-ink">Settings</h2>
        <p className="mt-0.5 text-xs text-ink-soft">
          Current backend configuration. Change these with environment variables
          and restart the service.
        </p>
      </div>

      <Panel>
        <PanelHeading
          title="Service"
          subtitle={health ? `${health.app} v${health.version}` : 'Backend unreachable'}
        />
        {health ? (
          <dl className="grid gap-x-8 gap-y-2 text-xs sm:grid-cols-2">
            <Row label="Status" value={health.status} env="" />
            <Row label="Default model" value={health.default_model} env="VISIONAI_DEFAULT_MODEL" />
            <Row label="Inference device" value={health.device.toUpperCase()} env="VISIONAI_DEVICE" />
            <Row
              label="Confidence threshold"
              value={String(health.defaults.confidence_threshold)}
              env="VISIONAI_CONFIDENCE_THRESHOLD"
            />
            <Row
              label="IoU threshold"
              value={String(health.defaults.iou_threshold)}
              env="VISIONAI_IOU_THRESHOLD"
            />
            <Row
              label="Max upload size"
              value={`${health.max_upload_mb} MB`}
              env="VISIONAI_MAX_UPLOAD_MB"
            />
            <Row
              label="Accepted formats"
              value={health.allowed_formats.join(', ').toUpperCase()}
              env=""
            />
            <Row
              label="History"
              value={health.history_enabled ? 'Enabled (SQLite)' : 'Disabled'}
              env="VISIONAI_HISTORY_ENABLED"
            />
            <Row
              label="ONNX providers"
              value={health.providers.join(', ') || 'none'}
              env=""
            />
            <Row
              label="Models in memory"
              value={health.loaded_models.join(', ') || 'none'}
              env="—"
            />
          </dl>
        ) : (
          <p className="text-xs text-ink-soft">
            Start the backend on port 8000 to see its configuration.
          </p>
        )}
      </Panel>

      <Panel>
        <PanelHeading
          title="Models"
          subtitle={
            models
              ? `${models.models.length} configured · at most ${models.max_loaded} held in memory at once`
              : 'Unavailable'
          }
        />
        <ul className="space-y-2">
          {(models?.models ?? []).map((model) => (
            <ModelRow key={model.name} model={model} />
          ))}
        </ul>
        <p className="mt-3 border-t border-edge pt-3 text-[11px] leading-relaxed text-ink-faint">
          Add a custom-trained model by pointing{' '}
          <code className="font-mono text-ink-soft">VISIONAI_MODELS_FILE</code> at a
          JSON registry. Class names are read from the checkpoint's own metadata,
          so no code change is needed.
        </p>
      </Panel>
    </div>
  )
}

function ModelRow({ model }: { model: ModelStatus }) {
  const icons: Record<string, ReactNode> = {
    loaded: <CheckCircle2 size={14} className="text-good" />,
    available: <CircleDashed size={14} className="text-warn" />,
    missing: <CircleAlert size={14} className="text-bad" />,
    error: <XCircle size={14} className="text-bad" />,
  }

  return (
    <li className="rounded-lg border border-edge px-3 py-2.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 shrink-0">{icons[model.status]}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs font-medium text-ink">{model.label}</p>
            <code className="rounded bg-canvas px-1.5 py-0.5 font-mono text-[10px] text-ink-soft">
              {model.name}
            </code>
            {model.is_default && (
              <span className="rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium text-accent">
                default
              </span>
            )}
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-ink-soft">
            {model.description}
          </p>
          <p className="mt-1 text-[11px] text-ink-faint">
            {model.detail}
            {model.class_count !== null && ` · ${model.class_count} classes`}
          </p>
        </div>
      </div>
    </li>
  )
}

function Row({ label, value, env }: { label: string; value: string; env: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-edge pb-2">
      <dt className="shrink-0 text-ink-faint">
        {label}
        {env && env !== '—' && (
          <code className="mt-0.5 block font-mono text-[10px] text-ink-faint/70">
            {env}
          </code>
        )}
      </dt>
      <dd className="truncate text-right font-medium text-ink">{value}</dd>
    </div>
  )
}
