import { useCallback, useEffect, useMemo, useState } from 'react'

import { MobileNav, Sidebar, type Page } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import type { SelectedImage } from './components/UploadPanel'
import { DashboardPage } from './pages/DashboardPage'
import { DetectionPage } from './pages/DetectionPage'
import { HistoryPage } from './pages/HistoryPage'
import { SettingsPage } from './pages/SettingsPage'
import { detect, getHealth, getModels, rerun } from './services/api'
import type {
  DetectionControls,
  DetectionResult,
  Health,
  ModelsResponse,
  ViewMode,
  ViewOptions,
} from './types'

const DEFAULT_CONTROLS: DetectionControls = {
  confidence_threshold: 0.35,
  iou_threshold: 0.45,
  max_detections: 300,
  model: 'yolo11n',
}

const DEFAULT_VIEW: ViewOptions = {
  show_boxes: true,
  show_labels: true,
  show_confidence: true,
}

/**
 * Application state lives here, one level above the pages.
 *
 * Two consequences worth the indirection: switching to History and back does
 * not discard the detection you are looking at, and there is exactly one place
 * where "what is currently visible" is decided -- which is what keeps the
 * cards, the charts, the table and the overlay from ever disagreeing.
 */
export default function App() {
  const [page, setPage] = useState<Page>('detect')

  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [checking, setChecking] = useState(true)
  const [models, setModels] = useState<ModelsResponse | null>(null)

  const [selected, setSelected] = useState<SelectedImage | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [controls, setControls] = useState<DetectionControls>(DEFAULT_CONTROLS)
  const [view, setView] = useState<ViewOptions>(DEFAULT_VIEW)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [mode, setMode] = useState<ViewMode>('annotated')
  const [settingsOpen, setSettingsOpen] = useState(true)
  const [historyToken, setHistoryToken] = useState(0)

  const checkBackend = useCallback(async () => {
    setChecking(true)
    try {
      const [healthResponse, modelsResponse] = await Promise.all([
        getHealth(),
        getModels(),
      ])
      setHealth(healthResponse)
      setModels(modelsResponse)
      setHealthError(null)

      // Adopt the server's own defaults, so the sliders start where the backend
      // actually is rather than at values hardcoded in the browser.
      setControls((current) =>
        current === DEFAULT_CONTROLS
          ? {
              confidence_threshold: healthResponse.defaults.confidence_threshold,
              iou_threshold: healthResponse.defaults.iou_threshold,
              max_detections: healthResponse.defaults.max_detections,
              model: healthResponse.default_model,
            }
          : current,
      )
    } catch (caught) {
      setHealth(null)
      setModels(null)
      setHealthError(
        caught instanceof Error ? caught.message : 'Could not reach the backend.',
      )
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    void checkBackend()
  }, [checkBackend])

  // Object URLs are revoked when the image is replaced, so a long session does
  // not accumulate the full bytes of every photograph ever opened.
  useEffect(() => {
    return () => {
      if (selected) URL.revokeObjectURL(selected.previewUrl)
    }
  }, [selected])

  const selectImage = useCallback(
    (image: SelectedImage) => {
      setSelected((previous) => {
        if (previous) URL.revokeObjectURL(previous.previewUrl)
        return image
      })
      setResult(null)
      setHidden(new Set())
      setSelectedId(null)
      setError(null)
    },
    [],
  )

  const clearImage = useCallback(() => {
    setSelected((previous) => {
      if (previous) URL.revokeObjectURL(previous.previewUrl)
      return null
    })
    setResult(null)
    setHidden(new Set())
    setSelectedId(null)
    setError(null)
  }, [])

  const runDetection = useCallback(async () => {
    if (!selected) return
    setBusy(true)
    setError(null)
    try {
      const response = await detect(selected.file, controls, view)
      setResult(response)
      setHidden(new Set())
      setSelectedId(null)
      setHistoryToken((token) => token + 1)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Detection failed.')
    } finally {
      setBusy(false)
    }
  }, [selected, controls, view])

  const applySettings = useCallback(async () => {
    if (!result) return
    setBusy(true)
    setError(null)
    try {
      const response = await rerun(result.result_id, controls, view)
      setResult(response)
      setSelectedId(null)
      setHistoryToken((token) => token + 1)
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'Could not apply the settings.',
      )
    } finally {
      setBusy(false)
    }
  }, [result, controls, view])

  /**
   * Changing the model adopts the threshold that model is worth running at.
   *
   * Open-vocabulary scores sit far below a closed-set model's -- a correct
   * detection at 0.20 is normal there, where on COCO it would be noise. Leaving
   * the slider at 0.35 after switching would show an empty result and look like
   * the model had failed.
   */
  const changeControls = useCallback(
    (next: DetectionControls) => {
      if (next.model !== controls.model) {
        const picked = models?.models.find((model) => model.name === next.model)
        if (picked?.recommended_confidence != null) {
          next = { ...next, confidence_threshold: picked.recommended_confidence }
        }
      }
      setControls(next)
    },
    [controls.model, models],
  )

  const resetSettings = useCallback(() => {
    setControls({
      confidence_threshold: health?.defaults.confidence_threshold ?? 0.35,
      iou_threshold: health?.defaults.iou_threshold ?? 0.45,
      max_detections: health?.defaults.max_detections ?? 300,
      model: health?.default_model ?? 'yolo11n',
    })
    setView(DEFAULT_VIEW)
  }, [health])

  /**
   * What is on screen.
   *
   * Raising the confidence slider filters immediately, because every detection
   * above the run's threshold is already in the browser. Lowering it past that
   * threshold cannot be done here -- those detections were never sent -- which
   * is what `needsRerun` tells the settings panel.
   */
  const visible = useMemo(() => {
    if (!result) return []
    return result.detections.filter(
      (detection) =>
        !hidden.has(detection.class_name) &&
        detection.confidence >= controls.confidence_threshold,
    )
  }, [result, hidden, controls.confidence_threshold])

  const visibleClasses = useMemo(() => {
    if (!result) return []
    const present = new Set(visible.map((detection) => detection.class_name))
    return result.classes.filter((entry) => present.has(entry.class_name))
  }, [result, visible])

  const needsRerun = useMemo(() => {
    if (!result) return false
    return (
      controls.model !== result.model ||
      controls.iou_threshold !== result.settings.iou_threshold ||
      controls.max_detections !== result.settings.max_detections ||
      controls.confidence_threshold < result.settings.confidence_threshold
    )
  }, [result, controls])

  const toggleClass = useCallback((className: string) => {
    setHidden((current) => {
      const next = new Set(current)
      if (next.has(className)) next.delete(className)
      else next.add(className)
      return next
    })
  }, [])

  const setAllClasses = useCallback(
    (allVisible: boolean) => {
      if (allVisible) {
        setHidden(new Set())
      } else {
        setHidden(new Set(result?.classes.map((entry) => entry.class_name) ?? []))
      }
    },
    [result],
  )

  return (
    <div className="flex h-full">
      <Sidebar
        page={page}
        onNavigate={setPage}
        detectionCount={result ? visible.length : null}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          health={health}
          checking={checking}
          error={healthError}
          onRefresh={() => void checkBackend()}
          onOpenSettings={() => setPage('settings')}
        />
        <MobileNav page={page} onNavigate={setPage} />

        <main className="flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-[1600px]">
            {page === 'dashboard' && (
              <DashboardPage
                health={health}
                result={result}
                onStart={() => setPage('detect')}
              />
            )}

            {page === 'detect' && (
              <DetectionPage
                health={health}
                models={models}
                selected={selected}
                result={result}
                visible={visible}
                visibleClasses={visibleClasses}
                hidden={hidden}
                controls={controls}
                view={view}
                mode={mode}
                selectedId={selectedId}
                busy={busy}
                error={error ?? healthError}
                settingsOpen={settingsOpen}
                needsRerun={needsRerun}
                onSelectImage={selectImage}
                onClearImage={clearImage}
                onDetect={() => void runDetection()}
                onApply={() => void applySettings()}
                onResetSettings={resetSettings}
                onControlsChange={changeControls}
                onViewChange={setView}
                onModeChange={setMode}
                onToggleClass={toggleClass}
                onSetAllClasses={setAllClasses}
                onSelectObject={setSelectedId}
                onToggleSettings={() => setSettingsOpen((open) => !open)}
                onError={setError}
                onDismissError={() => setError(null)}
              />
            )}

            {page === 'history' && <HistoryPage refreshToken={historyToken} />}

            {page === 'settings' && (
              <SettingsPage health={health} models={models} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
