const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10]

function chunkName(bytes, offset) {
  return String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  )
}

function joinBytes(chunks) {
  const size = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const result = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.length
  }
  return result
}

function parseLabelPng(buffer) {
  const bytes = new Uint8Array(buffer)
  if (bytes.length < PNG_SIGNATURE.length || !PNG_SIGNATURE.every((value, index) => bytes[index] === value)) return null

  let offset = PNG_SIGNATURE.length
  let header = null
  const compressedRows = []
  while (offset + 12 <= bytes.length) {
    const length = new DataView(bytes.buffer, bytes.byteOffset + offset, 4).getUint32(0)
    const dataStart = offset + 8
    const dataEnd = dataStart + length
    if (dataEnd + 4 > bytes.length) return null
    const name = chunkName(bytes, offset + 4)
    if (name === 'IHDR' && length === 13) {
      const view = new DataView(bytes.buffer, bytes.byteOffset + dataStart, length)
      header = {
        width: view.getUint32(0),
        height: view.getUint32(4),
        bitDepth: bytes[dataStart + 8],
        colorType: bytes[dataStart + 9],
        interlace: bytes[dataStart + 12],
      }
    } else if (name === 'IDAT') {
      compressedRows.push(bytes.slice(dataStart, dataEnd))
    }
    offset = dataEnd + 4
    if (name === 'IEND') break
  }
  if (!header || ![0, 2, 3].includes(header.colorType) || header.interlace !== 0 || !compressedRows.length) return null
  if (![1, 2, 4, 8].includes(header.bitDepth)) return null
  if (header.colorType === 2 && header.bitDepth !== 8) return null
  return { ...header, compressed: joinBytes(compressedRows) }
}

async function inflate(bytes) {
  if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decode PNG label data.')
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

function paeth(left, above, upperLeft) {
  const estimate = left + above - upperLeft
  const leftDistance = Math.abs(estimate - left)
  const aboveDistance = Math.abs(estimate - above)
  const upperLeftDistance = Math.abs(estimate - upperLeft)
  if (leftDistance <= aboveDistance && leftDistance <= upperLeftDistance) return left
  if (aboveDistance <= upperLeftDistance) return above
  return upperLeft
}

function unfilterRow(filtered, previous, filter, bytesPerPixel) {
  const row = new Uint8Array(filtered.length)
  for (let index = 0; index < row.length; index += 1) {
    const left = index >= bytesPerPixel ? row[index - bytesPerPixel] : 0
    const above = previous?.[index] ?? 0
    const upperLeft = index >= bytesPerPixel ? (previous?.[index - bytesPerPixel] ?? 0) : 0
    const value = filtered[index]
    if (filter === 0) row[index] = value
    else if (filter === 1) row[index] = (value + left) & 255
    else if (filter === 2) row[index] = (value + above) & 255
    else if (filter === 3) row[index] = (value + Math.floor((left + above) / 2)) & 255
    else if (filter === 4) row[index] = (value + paeth(left, above, upperLeft)) & 255
    else throw new Error(`Unsupported PNG row filter: ${filter}.`)
  }
  return row
}

function unpackRow(row, width, bitDepth) {
  if (bitDepth === 8) return row.slice(0, width)
  const mask = (1 << bitDepth) - 1
  const perByte = 8 / bitDepth
  const labels = new Uint8Array(width)
  for (let index = 0; index < width; index += 1) {
    const byte = row[Math.floor(index / perByte)]
    const shift = 8 - bitDepth * ((index % perByte) + 1)
    labels[index] = (byte >> shift) & mask
  }
  return labels
}

export async function decodePngLabels(buffer) {
  const png = parseLabelPng(buffer)
  if (!png) return null
  const channels = png.colorType === 2 ? 3 : 1
  const rowBytes = Math.ceil(png.width * channels * png.bitDepth / 8)
  const bytesPerPixel = Math.max(1, Math.ceil(channels * png.bitDepth / 8))
  const decoded = await inflate(png.compressed)
  const labels = new Uint8Array(png.width * png.height)
  let sourceOffset = 0
  let previous = null
  for (let rowIndex = 0; rowIndex < png.height; rowIndex += 1) {
    const filter = decoded[sourceOffset]
    const filtered = decoded.slice(sourceOffset + 1, sourceOffset + 1 + rowBytes)
    const row = unfilterRow(filtered, previous, filter, bytesPerPixel)
    const outputOffset = rowIndex * png.width
    if (png.colorType === 2) {
      for (let column = 0; column < png.width; column += 1) {
        const pixelOffset = column * 3
        const value = row[pixelOffset]
        if (value !== row[pixelOffset + 1] || value !== row[pixelOffset + 2]) {
          throw new Error('RGB PNG labels must use identical channel values.')
        }
        labels[outputOffset + column] = value
      }
    } else {
      labels.set(unpackRow(row, png.width, png.bitDepth), outputOffset)
    }
    previous = row
    sourceOffset += rowBytes + 1
  }
  return { width: png.width, height: png.height, labels }
}

export const decodeIndexedPng = decodePngLabels

export async function readPngLabels(file) {
  const name = file?.name?.toLowerCase() || ''
  if (file?.type !== 'image/png' && !name.endsWith('.png')) {
    throw new Error('Only PNG label images are supported.')
  }
  const labels = await decodePngLabels(await file.arrayBuffer())
  if (!labels) {
    throw new Error('PNG must contain raw grayscale, RGB, or indexed label values.')
  }
  return labels
}

export function cropAndResizeLabels(source, crop, inputSize) {
  const labels = []
  for (let row = 0; row < inputSize; row += 1) {
    const sourceY = Math.min(source.height - 1, Math.max(0, Math.floor(crop.y + (row + 0.5) * crop.size / inputSize)))
    const outputRow = []
    for (let column = 0; column < inputSize; column += 1) {
      const sourceX = Math.min(source.width - 1, Math.max(0, Math.floor(crop.x + (column + 0.5) * crop.size / inputSize)))
      outputRow.push(source.labels[sourceY * source.width + sourceX])
    }
    labels.push(outputRow)
  }
  return labels
}
