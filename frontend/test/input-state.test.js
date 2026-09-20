import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createRequestGate,
  normalizeBlocks,
  normalizeSeed,
} from '../src/input-state.js'
import { cropAndResizeLabels, decodePngLabels } from '../src/image-labels.js'

test('seed is always a nonnegative integer', () => {
  assert.equal(normalizeSeed(-4), 0)
  assert.equal(normalizeSeed(7.9), 7)
  assert.equal(normalizeSeed('12'), 12)
  assert.equal(normalizeSeed('invalid'), 0)
})

test('three block counts are clamped to the supported range', () => {
  assert.deepEqual(normalizeBlocks([0, 3.8, 99]), [1, 3, 4])
  assert.deepEqual(normalizeBlocks(), [1, 1, 1])
})

test('an input change rejects an in-flight response', () => {
  const gate = createRequestGate()
  const request = gate.begin()

  gate.invalidate()

  assert.equal(gate.accepts(request), false)
  assert.equal(gate.ownsRequest(request), true)
})

test('only the latest request may publish a result or clear busy state', () => {
  const gate = createRequestGate()
  const first = gate.begin()
  const second = gate.begin()

  assert.equal(gate.accepts(first), false)
  assert.equal(gate.ownsRequest(first), false)
  assert.equal(gate.accepts(second), true)
  assert.equal(gate.ownsRequest(second), true)
})

test('indexed PNG data is decoded as raw phase indices', async () => {
  const { deflateSync } = await import('node:zlib')
  const chunk = (name, data) => {
    const output = new Uint8Array(12 + data.length)
    const view = new DataView(output.buffer)
    view.setUint32(0, data.length)
    output.set(new TextEncoder().encode(name), 4)
    output.set(data, 8)
    return output
  }
  const bytes = new Uint8Array([
    137, 80, 78, 71, 13, 10, 26, 10,
  ])
  const header = new Uint8Array([0, 0, 0, 3, 0, 0, 0, 1, 8, 3, 0, 0, 0])
  const palette = new Uint8Array([239, 71, 111, 17, 138, 178, 6, 167, 125])
  const compressed = deflateSync(new Uint8Array([0, 0, 1, 2]))
  const png = new Uint8Array(bytes.length + 12 + header.length + 12 + palette.length + 12 + compressed.length + 12)
  let offset = 0
  for (const part of [bytes, chunk('IHDR', header), chunk('PLTE', palette), chunk('IDAT', compressed), chunk('IEND', new Uint8Array())]) {
    png.set(part, offset)
    offset += part.length
  }
  const decoded = await decodePngLabels(png.buffer)
  assert.deepEqual(Array.from(decoded.labels), [0, 1, 2])
})

test('grayscale PNG data is decoded as raw phase values', async () => {
  const { deflateSync } = await import('node:zlib')
  const chunk = (name, data) => {
    const output = new Uint8Array(12 + data.length)
    const view = new DataView(output.buffer)
    view.setUint32(0, data.length)
    output.set(new TextEncoder().encode(name), 4)
    output.set(data, 8)
    return output
  }
  const png = new Uint8Array([
    137, 80, 78, 71, 13, 10, 26, 10,
    ...chunk('IHDR', new Uint8Array([0, 0, 0, 3, 0, 0, 0, 1, 8, 0, 0, 0, 0])),
    ...chunk('IDAT', deflateSync(new Uint8Array([0, 0, 1, 2]))),
    ...chunk('IEND', new Uint8Array()),
  ])
  const decoded = await decodePngLabels(png.buffer)
  assert.deepEqual(Array.from(decoded.labels), [0, 1, 2])
})

test('indexed labels are resized with nearest-neighbor sampling', () => {
  const source = { width: 3, height: 1, labels: new Uint8Array([0, 1, 2]) }
  assert.deepEqual(cropAndResizeLabels(source, { x: 0, y: 0, size: 3 }, 6), Array.from({ length: 6 }, () => [0, 0, 1, 1, 2, 2]))
})
