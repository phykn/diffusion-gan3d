export const MIN_BLOCKS = 1
export const MAX_BLOCKS = 4

function integerInRange(value, minimum, maximum, fallback) {
  const number = Number(value)
  if (!Number.isFinite(number)) return fallback
  return Math.min(maximum, Math.max(minimum, Math.trunc(number)))
}

export function normalizeSeed(value) {
  return integerInRange(value, 0, Number.MAX_SAFE_INTEGER, 0)
}

export function normalizeBlock(value) {
  return integerInRange(value, MIN_BLOCKS, MAX_BLOCKS, MIN_BLOCKS)
}

export function normalizeBlocks(values) {
  return Array.from({ length: 3 }, (_, index) => normalizeBlock(values?.[index]))
}

export function createRequestGate() {
  let inputRevision = 0
  let latestRequest = 0

  return {
    invalidate() {
      inputRevision += 1
    },
    begin() {
      latestRequest += 1
      return { inputRevision, request: latestRequest }
    },
    accepts(token) {
      return token.inputRevision === inputRevision && token.request === latestRequest
    },
    ownsRequest(token) {
      return token.request === latestRequest
    },
  }
}
