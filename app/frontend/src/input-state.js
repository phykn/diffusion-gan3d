export const MIN_BLOCKS = 1

function integerInRange(value, minimum, maximum, fallback) {
  const number = Number(value)
  if (!Number.isFinite(number)) return fallback
  return Math.min(maximum, Math.max(minimum, Math.trunc(number)))
}

export function normalizeSeed(value) {
  return integerInRange(value, 0, Number.MAX_SAFE_INTEGER, 0)
}

// maximum is the server's per-axis limit from /api/health.
export function normalizeBlock(value, maximum) {
  return integerInRange(value, MIN_BLOCKS, maximum, MIN_BLOCKS)
}

export function normalizeBlocks(values, maximum) {
  return Array.from({ length: 3 }, (_, index) => normalizeBlock(values?.[index], maximum))
}

// Inputs changed after begin() make that request's result stale.
export function createRequestGate() {
  let revision = 0

  return {
    invalidate() {
      revision += 1
    },
    begin() {
      return revision
    },
    accepts(token) {
      return token === revision
    },
  }
}
