import { ScanSearch } from 'lucide-react'

import { ClassFilter } from '../components/ClassFilter'
import { DetectionTable } from '../components/DetectionTable'
import { DistributionCharts } from '../components/DistributionCharts'
import { ExportBar } from '../components/ExportBar'
import { ImageViewer } from '../components/ImageViewer'
import { SettingsPanel } from '../components/SettingsPanel'
import { SummaryCards } from '../components/SummaryCards'
import { UploadPanel, type SelectedImage } from '../components/UploadPanel'
import { EmptyState, ErrorNotice, Panel, PanelHeading } from '../components/ui'
import type {
  ClassSummary,
  Detection,
  DetectionControls,
  DetectionResult,
  Health,
  ModelsResponse,
  ViewMode,
  ViewOptions,
} from '../types'

export function DetectionPage({
  health,
  models,
  selected,
  result,
  visible,
  visibleClasses,
  hidden,
  controls,
  view,
  mode,
  selectedId,
  busy,
  error,
  settingsOpen,
  needsRerun,
  onSelectImage,
  onClearImage,
  onDetect,
  onApply,
  onResetSettings,
  onControlsChange,
  onViewChange,
  onModeChange,
  onToggleClass,
  onSetAllClasses,
  onSelectObject,
  onToggleSettings,
  onError,
  onDismissError,
}: {
  health: Health | null
  models: ModelsResponse | null
  selected: SelectedImage | null
  result: DetectionResult | null
  visible: Detection[]
  visibleClasses: ClassSummary[]
  hidden: Set<string>
  controls: DetectionControls
  view: ViewOptions
  mode: ViewMode
  selectedId: number | null
  busy: boolean
  error: string | null
  settingsOpen: boolean
  needsRerun: boolean
  onSelectImage: (image: SelectedImage) => void
  onClearImage: () => void
  onDetect: () => void
  onApply: () => void
  onResetSettings: () => void
  onControlsChange: (controls: DetectionControls) => void
  onViewChange: (view: ViewOptions) => void
  onModeChange: (mode: ViewMode) => void
  onToggleClass: (className: string) => void
  onSetAllClasses: (allVisible: boolean) => void
  onSelectObject: (objectId: number | null) => void
  onToggleSettings: () => void
  onError: (message: string) => void
  onDismissError: () => void
}) {
  const allVisible = hidden.size === 0

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-ink">
          AI Object Detection
        </h2>
        <p className="mt-0.5 text-xs text-ink-soft">
          Upload an image to identify, locate and count the objects in it. Every
          figure below comes from the model's predictions on your image.
        </p>
      </div>

      {error && <ErrorNotice message={error} onDismiss={onDismissError} />}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_20rem]">
        {/* Primary column: the image stays the largest thing on the page. */}
        <div className="space-y-5">
          <Panel>
            <PanelHeading
              title="Upload image"
              subtitle={
                selected
                  ? 'Run detection, or clear the image to upload another'
                  : 'Drag, paste or browse for a photograph'
              }
            />
            <UploadPanel
              selected={selected}
              onSelect={onSelectImage}
              onClear={onClearImage}
              onDetect={onDetect}
              busy={busy}
              disabled={!health?.model_ready}
              maxUploadMb={health?.max_upload_mb ?? 15}
              onError={onError}
            />
          </Panel>

          {result && selected ? (
            <Panel className="min-h-[28rem]">
              <ImageViewer
                imageUrl={selected.previewUrl}
                image={result.image}
                detections={visible}
                view={view}
                mode={mode}
                onModeChange={onModeChange}
                selectedId={selectedId}
                onSelect={onSelectObject}
              />
            </Panel>
          ) : (
            <Panel>
              <EmptyState
                icon={<ScanSearch size={22} />}
                title="No detection yet"
                description="Upload an image and press Detect Objects. The annotated result, statistics and a table of every detected object will appear here."
              />
            </Panel>
          )}
        </div>

        {/* Secondary column: controls and per-class visibility. */}
        <div className="space-y-4">
          <SettingsPanel
            open={settingsOpen}
            onToggle={onToggleSettings}
            controls={controls}
            onControlsChange={onControlsChange}
            view={view}
            onViewChange={onViewChange}
            models={models}
            result={result}
            busy={busy}
            onApply={onApply}
            onReset={onResetSettings}
            dirty={needsRerun}
          />

          {result && (
            <ClassFilter
              classes={result.classes}
              hidden={hidden}
              onToggle={onToggleClass}
              onSetAll={onSetAllClasses}
            />
          )}

          {result && (
            <ExportBar
              result={result}
              view={view}
              visibleClasses={visibleClasses.map((entry) => entry.class_name)}
              allVisible={allVisible}
              minConfidence={controls.confidence_threshold}
            />
          )}
        </div>
      </div>

      {result && (
        <>
          <SummaryCards
            detections={visible}
            image={result.image}
            processingMs={result.processing_time_ms}
            inferenceMs={result.inference_time_ms}
            device={result.model_info.device}
          />
          <DistributionCharts detections={visible} />
          <DetectionTable
            detections={visible}
            selectedId={selectedId}
            onSelect={onSelectObject}
          />
        </>
      )}
    </div>
  )
}
