const PNG = [137, 80, 78, 71, 13, 10, 26, 10]
const SIZE_ERROR = 'PNG pixel data does not match its declared dimensions.'

function readChunks(buffer) {
  const bytes = new Uint8Array(buffer)
  if (!PNG.every((value, i) => bytes[i] === value)) return null

  const view = new DataView(buffer)
  const data = []
  let info
  for (let offset = PNG.length; offset + 12 <= bytes.length;) {
    const length = view.getUint32(offset)
    const start = offset + 8
    const end = start + length
    if (end + 4 > bytes.length) return null
    switch (view.getUint32(offset + 4)) {
      case 0x49484452: {
        if (length !== 13) return null
        info = {
          width: view.getUint32(start),
          height: view.getUint32(start + 4),
          bits: bytes[start + 8],
          color: bytes[start + 9],
          interlaced: bytes[start + 12] !== 0,
        }
        break
      }
      case 0x49444154:
        data.push(bytes.subarray(start, end))
        break
      case 0x49454e44:
        return info && data.length ? { ...info, data } : null
    }
    offset = end + 4
  }
  return null
}

async function inflateRows(parts, length) {
  if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decode PNG label data.')
  const reader = new Blob(parts).stream().pipeThrough(new DecompressionStream('deflate')).getReader()
  const chunks = []
  let bytes = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.length
      if (bytes > length) throw new Error(SIZE_ERROR)
      chunks.push(value)
    }
    if (bytes !== length) throw new Error(SIZE_ERROR)
    const result = new Uint8Array(bytes)
    let offset = 0
    for (const chunk of chunks) {
      result.set(chunk, offset)
      offset += chunk.length
    }
    return result
  } finally {
    try {
      await reader.cancel()
    } finally {
      reader.releaseLock()
    }
  }
}

function paeth(a, b, c) {
  const distanceA = Math.abs(b - c)
  const distanceB = Math.abs(a - c)
  const distanceC = Math.abs(a + b - 2 * c)
  if (distanceA <= distanceB && distanceA <= distanceC) return a
  return distanceB <= distanceC ? b : c
}

function restoreRow(row, above, filter, stride) {
  if (filter > 4) throw new Error(`Unsupported PNG row filter: ${filter}.`)
  if (!filter) return
  for (let i = 0; i < row.length; i++) {
    const left = i >= stride ? row[i - stride] : 0
    const top = above?.[i] ?? 0
    const diagonal = above?.[i - stride] ?? 0
    let prediction = left
    if (filter === 2) prediction = top
    if (filter === 3) prediction = (left + top) >> 1
    if (filter === 4) prediction = paeth(left, top, diagonal)
    row[i] += prediction
  }
}

export async function decodePngLabels(buffer) {
  const png = readChunks(buffer)
  if (!png || png.interlaced || ![0, 2, 3].includes(png.color)) return null
  const { width, height, bits, data, color } = png
  if (![1, 2, 4, 8].includes(bits) || (color === 2 && bits !== 8)) return null

  const channels = color === 2 ? 3 : 1
  const rowSize = Math.ceil(width * channels * bits / 8)
  const length = (rowSize + 1) * height
  if (!width || !height || !Number.isSafeInteger(length)) throw new Error(SIZE_ERROR)

  const rows = await inflateRows(data, length)
  const labels = new Uint8Array(width * height)
  let previous
  for (let y = 0; y < height; y++) {
    const start = y * (rowSize + 1)
    const row = rows.subarray(start + 1, start + 1 + rowSize)
    restoreRow(row, previous, rows[start], channels)
    previous = row
    if (channels === 1 && bits === 8) {
      labels.set(row, y * width)
      continue
    }
    for (let x = 0; x < width; x++) {
      if (channels === 3) {
        const i = 3 * x
        if (row[i] !== row[i + 1] || row[i] !== row[i + 2]) {
          throw new Error('RGB PNG labels must use identical channel values.')
        }
        labels[y * width + x] = row[i]
      } else {
        const bit = x * bits
        labels[y * width + x] = (row[Math.floor(bit / 8)] >> (8 - bits - (bit % 8))) & ((1 << bits) - 1)
      }
    }
  }
  return { width, height, labels }
}

export async function readPngLabels(file) {
  if (file?.type !== 'image/png' && !file?.name?.toLowerCase().endsWith('.png')) {
    throw new Error('Only PNG label images are supported.')
  }
  const source = await decodePngLabels(await file.arrayBuffer())
  if (!source) throw new Error('PNG must contain raw grayscale, RGB, or indexed label values.')
  return source
}

export function clampCrop(source, crop, size) {
  const bound = (position, edge) => Math.max(0, Math.min(Math.round(position), edge - size))
  return { x: bound(crop.x, source.width), y: bound(crop.y, source.height) }
}

export function cropAnchor(source, crop, size) {
  const { width, height, labels } = source
  const { x, y } = clampCrop(source, crop, size)
  const image = []
  for (let row = y; row < y + size; row++) {
    const start = row * width + x
    image.push([...labels.subarray(start, start + size)])
  }
  return { image, cropOrigin: [y, x], sourceShape: [height, width] }
}
