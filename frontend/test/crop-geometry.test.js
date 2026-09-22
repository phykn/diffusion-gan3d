import assert from 'node:assert/strict'
import test from 'node:test'
import { cropAnchor } from '../src/image-labels.js'

test('crop labels and physical origin share the same pixel boundaries', () => {
  const source = { width: 8, height: 10, labels: Uint8Array.from({ length: 80 }, (_, i) => i) }
  const crop = cropAnchor(source, { x: 1.6, y: 3.7 }, 2)
  assert.deepEqual(crop, { image: [[34, 35], [42, 43]], cropOrigin: [4, 2], sourceShape: [10, 8] })
})

test('a homogeneous measured section remains a valid anchor', () => {
  const crop = cropAnchor({ width: 2, height: 2, labels: new Uint8Array(4) }, { x: 0, y: 0 }, 2)
  assert.deepEqual(crop.image, [[0, 0], [0, 0]])
})
