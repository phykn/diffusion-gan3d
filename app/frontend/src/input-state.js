export const MIN_BLOCKS = 1

function readInteger(value, min, max) {
  const integer = Math.trunc(Number(value))
  return Number.isFinite(integer) ? Math.max(min, Math.min(integer, max)) : min
}

export function normalizeSeed(value) {
  return readInteger(value, 0, Number.MAX_SAFE_INTEGER)
}

export function normalizeBlock(value, maximum) {
  return readInteger(value, MIN_BLOCKS, maximum)
}

export function normalizeBlocks(values, maximum) {
  return [0, 1, 2].map(axis => normalizeBlock(values?.[axis], maximum))
}

export function anchorNote(blocks, cropSize) {
  return blocks?.some(count => count > 1)
    ? `Anchor covers the center ${cropSize}×${cropSize} of the first section`
    : null
}
