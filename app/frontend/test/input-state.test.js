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

test('three block counts are clamped to the server limit', () => {
  assert.deepEqual(normalizeBlocks([0, 3.8, 99], 4), [1, 3, 4])
  assert.deepEqual(normalizeBlocks([0, 3.8, 99], 9), [1, 3, 9])
  assert.deepEqual(normalizeBlocks(undefined, 4), [1, 1, 1])
})

test('an input change rejects an in-flight response', () => {
  const gate = createRequestGate()
  const request = gate.begin()
  assert.equal(gate.accepts(request), true)

  gate.invalidate()

  assert.equal(gate.accepts(request), false)
  assert.equal(gate.accepts(gate.begin()), true)
})
