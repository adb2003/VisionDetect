import { useCallback, useEffect, useRef, useState } from 'react'
import { ImageUp, Play, Trash2, UploadCloud } from 'lucide-react'

import { bytes, classNames } from '../lib/format'
import { Button, Spinner } from './ui'

export interface SelectedImage {
  file: File
  previewUrl: string
  width: number
  height: number
}

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const ACCEPT_ATTRIBUTE = '.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp'

/**
 * Validates in the browser before anything is sent.
 *
 * The backend validates again and is the real authority -- these checks exist
 * to fail in a tenth of a second instead of after a multi-megabyte upload.
 */
function validate(file: File, maxMb: number): string | null {
  const extension = file.name.split('.').pop()?.toLowerCase() ?? ''
  const typeOk = ACCEPTED.includes(file.type)
  const extensionOk = ['jpg', 'jpeg', 'png', 'webp'].includes(extension)

  if (!typeOk && !extensionOk) {
    return `${file.name} is not a supported image. Use JPG, PNG or WEBP.`
  }
  if (file.size === 0) {
    return `${file.name} is empty.`
  }
  if (file.size > maxMb * 1024 * 1024) {
    return `${file.name} is ${bytes(file.size)}, above the ${maxMb} MB limit.`
  }
  return null
}

export function UploadPanel({
  selected,
  onSelect,
  onClear,
  onDetect,
  busy,
  disabled,
  maxUploadMb,
  onError,
}: {
  selected: SelectedImage | null
  onSelect: (image: SelectedImage) => void
  onClear: () => void
  onDetect: () => void
  busy: boolean
  disabled: boolean
  maxUploadMb: number
  onError: (message: string) => void
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return
      const problem = validate(file, maxUploadMb)
      if (problem) {
        onError(problem)
        return
      }

      // Read the real pixel dimensions before showing anything, so the panel
      // never displays a size it is guessing at.
      const url = URL.createObjectURL(file)
      const probe = new Image()
      probe.onload = () => {
        onSelect({
          file,
          previewUrl: url,
          width: probe.naturalWidth,
          height: probe.naturalHeight,
        })
      }
      probe.onerror = () => {
        URL.revokeObjectURL(url)
        onError(`${file.name} could not be read as an image. It may be corrupt.`)
      }
      probe.src = url
    },
    [maxUploadMb, onError, onSelect],
  )

  // Paste-to-upload: the quickest path from a screenshot to a detection.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const file = Array.from(event.clipboardData?.files ?? [])[0]
      if (file) accept(file)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [accept])

  return (
    <div>
      {!selected ? (
        <div
          onDragEnter={(event) => {
            event.preventDefault()
            dragDepth.current += 1
            setDragging(true)
          }}
          onDragLeave={(event) => {
            event.preventDefault()
            dragDepth.current -= 1
            if (dragDepth.current <= 0) setDragging(false)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            dragDepth.current = 0
            setDragging(false)
            accept(event.dataTransfer.files[0])
          }}
          className={classNames(
            'rounded-xl border-2 border-dashed px-6 py-10 text-center transition',
            dragging
              ? 'border-accent bg-accent-soft'
              : 'border-edge-strong bg-canvas hover:border-ink-faint',
          )}
        >
          <div className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-full bg-panel text-ink-soft ring-1 ring-edge">
            <UploadCloud size={19} />
          </div>
          <p className="text-sm font-medium text-ink">
            Drag an image here, or paste from the clipboard
          </p>
          <p className="mt-1 text-xs text-ink-soft">
            JPG, PNG or WEBP · up to {maxUploadMb} MB
          </p>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT_ATTRIBUTE}
            className="hidden"
            onChange={(event) => {
              accept(event.target.files?.[0])
              // Reset so re-picking the same file fires change again.
              event.target.value = ''
            }}
          />
          <Button
            variant="primary"
            className="mt-4"
            onClick={() => inputRef.current?.click()}
          >
            <ImageUp size={14} />
            Browse files
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-start gap-4 rounded-xl border border-edge bg-canvas p-3">
            <img
              src={selected.previewUrl}
              alt={`Preview of ${selected.file.name}`}
              className="h-16 w-16 shrink-0 rounded-lg border border-edge object-cover"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium text-ink" title={selected.file.name}>
                {selected.file.name}
              </p>
              <dl className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-ink-soft">
                <div className="flex gap-1">
                  <dt className="text-ink-faint">Size</dt>
                  <dd className="font-mono tabular-nums">{bytes(selected.file.size)}</dd>
                </div>
                <div className="flex gap-1">
                  <dt className="text-ink-faint">Dimensions</dt>
                  <dd className="font-mono tabular-nums">
                    {selected.width} × {selected.height}
                  </dd>
                </div>
                <div className="flex gap-1">
                  <dt className="text-ink-faint">Type</dt>
                  <dd className="uppercase">
                    {selected.file.name.split('.').pop() ?? 'image'}
                  </dd>
                </div>
              </dl>
            </div>
            <Button variant="ghost" onClick={onClear} title="Remove this image">
              <Trash2 size={14} />
            </Button>
          </div>

          <Button
            variant="primary"
            onClick={onDetect}
            disabled={busy || disabled}
            className="w-full py-2.5 text-sm"
            title={
              disabled ? 'The backend is not ready' : 'Run the detection model'
            }
          >
            {busy ? (
              <>
                <Spinner className="h-4 w-4" />
                Detecting objects…
              </>
            ) : (
              <>
                <Play size={15} />
                Detect Objects
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}
