/**
 * Backend client.
 *
 * All paths are relative, which the Vite dev server proxies to the backend.
 * That keeps `fetch` calls and plain `<a download>` links on the same origin,
 * so downloads work without blobs or CORS exposure of headers.
 */

import type {
  DetectionControls,
  DetectionResult,
  Health,
  HistoryResponse,
  ModelsResponse,
  ViewOptions,
} from '../types'

const API = '/api'

/** An error carrying the backend's own explanation, not a generic string. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>

  // FastAPI puts the message in `detail`, either as a string or, for request
  // validation failures, as a list of field errors.
  let detail = `Request failed with status ${response.status}`
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') {
      detail = body.detail
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail
        .map((item: { loc?: string[]; msg?: string }) => {
          const field = item.loc?.slice(1).join('.') ?? 'request'
          return `${field}: ${item.msg ?? 'invalid'}`
        })
        .join('; ')
    }
  } catch {
    // A non-JSON error body (a proxy failure, say) leaves the default message.
  }
  throw new ApiError(detail, response.status)
}

function networkError(error: unknown): never {
  if (error instanceof ApiError) throw error
  throw new ApiError(
    'Could not reach the backend. Check that it is running on port 8000 ' +
      '(see the README for the start command).',
    0,
  )
}

export async function getHealth(): Promise<Health> {
  try {
    return await unwrap<Health>(await fetch(`${API}/health`))
  } catch (error) {
    return networkError(error)
  }
}

export async function getModels(): Promise<ModelsResponse> {
  try {
    return await unwrap<ModelsResponse>(await fetch(`${API}/models`))
  } catch (error) {
    return networkError(error)
  }
}

export async function detect(
  file: File,
  controls: DetectionControls,
  view: ViewOptions,
): Promise<DetectionResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('confidence_threshold', String(controls.confidence_threshold))
  form.append('iou_threshold', String(controls.iou_threshold))
  form.append('max_detections', String(controls.max_detections))
  form.append('model', controls.model)
  form.append('show_boxes', String(view.show_boxes))
  form.append('show_labels', String(view.show_labels))
  form.append('show_confidence', String(view.show_confidence))

  try {
    return await unwrap<DetectionResult>(
      await fetch(`${API}/detect`, { method: 'POST', body: form }),
    )
  } catch (error) {
    return networkError(error)
  }
}

/**
 * Re-run on an image the server still holds.
 *
 * Needed whenever the threshold moves down: detections below the previous
 * threshold were never sent to the browser, so they cannot be recovered by
 * filtering what is already here.
 */
export async function rerun(
  resultId: string,
  controls: DetectionControls,
  view: ViewOptions,
): Promise<DetectionResult> {
  try {
    return await unwrap<DetectionResult>(
      await fetch(`${API}/results/${resultId}/rerun`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...controls, ...view }),
      }),
    )
  } catch (error) {
    return networkError(error)
  }
}

export async function getHistory(limit = 100): Promise<HistoryResponse> {
  try {
    return await unwrap<HistoryResponse>(await fetch(`${API}/history?limit=${limit}`))
  } catch (error) {
    return networkError(error)
  }
}

export async function deleteHistoryEntry(analysisId: string): Promise<void> {
  await fetch(`${API}/history/${analysisId}`, { method: 'DELETE' })
}

export async function clearHistory(): Promise<void> {
  await fetch(`${API}/history`, { method: 'DELETE' })
}

/**
 * URL for the annotated image, re-rendered server-side to match what is on
 * screen: the same class filter and the same label switches.
 */
export function annotatedImageUrl(
  resultId: string,
  options: {
    format?: 'png' | 'jpeg'
    classes?: string[] | null
    minConfidence?: number
    view?: ViewOptions
    download?: boolean
  } = {},
): string {
  const params = new URLSearchParams()
  params.set('format', options.format ?? 'png')
  if (options.classes) params.set('classes', options.classes.join(','))
  if (options.minConfidence) {
    params.set('min_confidence', String(options.minConfidence))
  }
  if (options.view) {
    params.set('boxes', String(options.view.show_boxes))
    params.set('labels', String(options.view.show_labels))
    params.set('scores', String(options.view.show_confidence))
  }
  if (options.download) params.set('download', 'true')
  return `${API}/results/${resultId}/image?${params.toString()}`
}

export const exportUrls = {
  json: (resultId: string) => `${API}/results/${resultId}/json`,
  csv: (resultId: string) => `${API}/results/${resultId}/csv`,
  report: (resultId: string) => `${API}/results/${resultId}/report`,
}
