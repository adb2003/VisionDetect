import { FileCode2, FileSpreadsheet, FileText, ImageDown } from 'lucide-react'
import type { ReactNode } from 'react'

import { annotatedImageUrl, exportUrls } from '../services/api'
import type { DetectionResult, ViewOptions } from '../types'
import { Panel, PanelHeading } from './ui'

/**
 * Downloads.
 *
 * Plain anchors rather than fetch-and-blob: the browser streams the file
 * straight from the API and uses the filename the server sets, and nothing has
 * to be held in memory on the way through.
 *
 * The annotated image is requested with the filters currently applied, so what
 * downloads is what is on screen rather than the unfiltered original run.
 */
export function ExportBar({
  result,
  view,
  visibleClasses,
  allVisible,
  minConfidence,
}: {
  result: DetectionResult
  view: ViewOptions
  visibleClasses: string[]
  allVisible: boolean
  minConfidence: number
}) {
  const filtered = !allVisible || minConfidence > result.settings.confidence_threshold

  const imageHref = annotatedImageUrl(result.result_id, {
    format: 'png',
    classes: allVisible ? null : visibleClasses,
    minConfidence,
    view,
    download: true,
  })

  return (
    <Panel>
      <PanelHeading
        title="Export"
        subtitle={
          filtered
            ? 'The image matches the filtered view; data files hold the full run'
            : 'Everything from this detection run'
        }
      />
      <div className="grid grid-cols-2 gap-2">
        <DownloadLink
          href={imageHref}
          icon={<ImageDown size={14} />}
          label="Annotated PNG"
          hint="Image with boxes"
        />
        <DownloadLink
          href={exportUrls.json(result.result_id)}
          icon={<FileCode2 size={14} />}
          label="JSON"
          hint="Full structured results"
        />
        <DownloadLink
          href={exportUrls.csv(result.result_id)}
          icon={<FileSpreadsheet size={14} />}
          label="CSV"
          hint="Detection table"
        />
        <DownloadLink
          href={exportUrls.report(result.result_id)}
          icon={<FileText size={14} />}
          label="Report"
          hint="Summary text"
        />
      </div>
    </Panel>
  )
}

function DownloadLink({
  href,
  icon,
  label,
  hint,
}: {
  href: string
  icon: ReactNode
  label: string
  hint: string
}) {
  return (
    <a
      href={href}
      download
      className="flex items-start gap-2.5 rounded-lg border border-edge px-3 py-2.5 transition hover:border-accent hover:bg-accent-soft"
    >
      <span className="mt-0.5 text-ink-soft">{icon}</span>
      <span className="min-w-0">
        <span className="block truncate text-xs font-medium text-ink">{label}</span>
        <span className="block truncate text-[11px] text-ink-faint">{hint}</span>
      </span>
    </a>
  )
}
