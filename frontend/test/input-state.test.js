import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createRequestGate,
  normalizeBlocks,
  normalizeSeed,
} from '../src/input-state.js'

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
