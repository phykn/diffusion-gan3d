import assert from 'node:assert/strict'
import test from 'node:test'
import { deflateSync } from 'node:zlib'

import { cropAndResizeLabels, decodePngLabels } from '../src/image-labels.js'

function pngChunk(name, data) {
  const output = new Uint8Array(12 + data.length)
  const view = new DataView(output.buffer)
  view.setUint32(0, data.length)
  output.set(new TextEncoder().encode(name), 4)
  output.set(data, 8)
  return output
}

test('indexed PNG data is decoded as raw phase indices', async () => {
  const bytes = new Uint8Array([
    137, 80, 78, 71, 13, 10, 26, 10,
  ])
  const header = new Uint8Array([0, 0, 0, 3, 0, 0, 0, 1, 8, 3, 0, 0, 0])
  const palette = new Uint8Array([239, 71, 111, 17, 138, 178, 6, 167, 125])
  const compressed = deflateSync(new Uint8Array([0, 0, 1, 2]))
  const png = new Uint8Array(bytes.length + 12 + header.length + 12 + palette.length + 12 + compressed.length + 12)
  let offset = 0
  for (const part of [bytes, pngChunk('IHDR', header), pngChunk('PLTE', palette), pngChunk('IDAT', compressed), pngChunk('IEND', new Uint8Array())]) {
    png.set(part, offset)
    offset += part.length
  }
  const decoded = await decodePngLabels(png.buffer)
  assert.deepEqual(Array.from(decoded.labels), [0, 1, 2])
})

test('grayscale PNG data is decoded as raw phase values', async () => {
  const png = new Uint8Array([
    137, 80, 78, 71, 13, 10, 26, 10,
    ...pngChunk('IHDR', new Uint8Array([0, 0, 0, 3, 0, 0, 0, 1, 8, 0, 0, 0, 0])),
    ...pngChunk('IDAT', deflateSync(new Uint8Array([0, 0, 1, 2]))),
    ...pngChunk('IEND', new Uint8Array()),
  ])
  const decoded = await decodePngLabels(png.buffer)
  assert.deepEqual(Array.from(decoded.labels), [0, 1, 2])
  // Increasing width without adding a pixel used to silently append phase 0.
  const truncated = png.slice()
  new DataView(truncated.buffer).setUint32(16, 4)
  await assert.rejects(decodePngLabels(truncated.buffer), /declared dimensions/)
})

test('indexed labels are resized with nearest-neighbor sampling', () => {
  const source = { width: 3, height: 1, labels: new Uint8Array([0, 1, 2]) }
  assert.deepEqual(cropAndResizeLabels(source, { x: 0, y: 0, size: 3 }, 6), Array.from({ length: 6 }, () => [0, 0, 1, 1, 2, 2]))
})
