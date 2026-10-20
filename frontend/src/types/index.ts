/**
 * Mirrors the backend's response models (`backend/app/schemas.py`).
 *
 * Field names are snake_case here rather than being converted to camelCase on
 * the way in: the shapes stay literally identical to the JSON the API returns
 * and to the JSON export, so there is one vocabulary to learn instead of two.
 */

export interface BoundingBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
}

export interface Point {
  x: number
  y: number
}

export interface Detection {
  object_id: number
  class_id: number
  class_name: string
  confidence: number
  bbox: BoundingBox
  width: number
  height: number
  center: Point
  area: number
  /** Supplied by the server so overlay and annotated image agree on colour. */
  color: string
}

export interface ClassSummary {
  class_id: number
  class_name: string
  count: number
  color: string
  avg_confidence: number
  max_confidence: number
}

export interface ImageMeta {
  filename: string
  width: number
  height: number
  format: string
  size_bytes: number
}

export interface DetectionSummary {
  total_objects: number
  unique_classes: number
  max_confidence: number
  min_confidence: number
  avg_confidence: number
  class_counts: Record<string, number>
}

export interface DetectionSettings {
  confidence_threshold: number
  iou_threshold: number
  max_detections: number
}

export interface ModelMeta {
  name: string
  label: string
  backend: string
  task: string
  device: string
  input_size: number
  class_count: number
}

export interface DetectionResult {
  result_id: string
  created_at: string
  image: ImageMeta
  model: string
  model_info: ModelMeta
  settings: DetectionSettings
  processing_time_ms: number
  inference_time_ms: number
  summary: DetectionSummary
  classes: ClassSummary[]
  detections: Detection[]
  annotated_image: string | null
  annotated_image_url: string
}

export interface ModelStatus {
  name: string
  label: string
  description: string
  backend: string
  task: string | null
  device: string | null
  input_size: number | null
  status: 'loaded' | 'available' | 'missing' | 'error'
  detail: string
  class_count: number | null
  classes: string[]
  weights_path: string
  is_default: boolean
  /** Threshold this model is worth running at; adopted when it is selected. */
  recommended_confidence: number | null
}

export interface ModelsResponse {
  models: ModelStatus[]
  default_model: string
  loaded: string[]
  max_loaded: number
}

export interface Health {
  status: 'ok' | 'degraded'
  app: string
  version: string
  model_ready: boolean
  model_status: string
  default_model: string
  device: string
  providers: string[]
  loaded_models: string[]
  history_enabled: boolean
  max_upload_mb: number
  allowed_formats: string[]
  defaults: DetectionSettings
}

export interface HistoryEntry {
  analysis_id: string
  filename: string
  timestamp: string
  total_objects: number
  unique_classes: number
  processing_time_ms: number
  model: string
  width: number
  height: number
  confidence_threshold: number
  iou_threshold: number
  top_classes: string
  result_available: boolean
}

export interface HistoryResponse {
  entries: HistoryEntry[]
  total: number
}

/** Settings the user controls in the panel, before they are applied. */
export interface DetectionControls {
  confidence_threshold: number
  iou_threshold: number
  max_detections: number
  model: string
}

/** Purely visual switches; these never require re-running the model. */
export interface ViewOptions {
  show_boxes: boolean
  show_labels: boolean
  show_confidence: boolean
}

export type ViewMode = 'annotated' | 'original' | 'split'
