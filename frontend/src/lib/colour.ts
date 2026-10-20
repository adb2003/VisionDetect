/**
 * Readability helper for the class colours the backend assigns.
 *
 * Mirrors `_contrasting_text_colour` in `backend/app/image_utils.py`: the
 * palette is bright enough that white text disappears on several of its hues,
 * so a label picks black or white by luminance. Both renderers use the same
 * rule, which is what keeps the on-screen overlay and the downloaded PNG
 * looking like the same artefact.
 */
export function contrastText(hex: string): string {
  const value = hex.replace('#', '')
  if (value.length !== 6) return '#ffffff'

  const red = parseInt(value.slice(0, 2), 16)
  const green = parseInt(value.slice(2, 4), 16)
  const blue = parseInt(value.slice(4, 6), 16)
  if ([red, green, blue].some(Number.isNaN)) return '#ffffff'

  const luminance = 0.299 * red + 0.587 * green + 0.114 * blue
  return luminance > 150 ? '#000000' : '#ffffff'
}
