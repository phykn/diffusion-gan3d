import assert from 'node:assert/strict'
import test from 'node:test'
import { generateVolume, prepareImage } from '../src/api.js'

test('generation sends a prepared boundary anchor and decodes volume metadata', async t => {
  const image = [[[0.25]], [[0.75]]]
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/generate')
    assert.deepEqual(JSON.parse(options.body), {
      anchors: [{ image, axis: 0, index: 0 }],
      seed: 7,
      blocks: [2, 1, 1],
      format: 'raw',
      include_metrics: true,
    })
    return new Response(new Uint8Array([0, 1]), { headers: {
      'X-Volume-Shape': '2,1,1', 'X-Porosity': '0.5', 'X-Tortuosity': 'unavailable',
    } })
  })
  const result = await generateVolume(image, 7, [2, 1, 1])
  assert.deepEqual(result, { shape: [2, 1, 1], values: new Uint8Array([0, 1]), porosity: 0.5, tortuosity: null })
})

test('single-block requests omit tiled geometry', async t => {
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal('blocks' in JSON.parse(options.body), false)
    return new Response(new Uint8Array([0]), { headers: { 'X-Volume-Shape': '1,1,1' } })
  })
  await generateVolume([[0]], 0, [1, 1, 1])
})

for (const shape of ['-1,-1,1', '0,1,1', '0.5,2,1', 'NaN,1,1', '1,1', '']) {
  test(`reject invalid volume dimensions: ${shape}`, async t => {
    t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([0]), {
      headers: { 'X-Volume-Shape': shape },
    }))
    await assert.rejects(generateVolume([[0]], 0, [1, 1, 1]), /invalid volume shape/)
  })
}

test('reject truncated volume data', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([0]), {
    headers: { 'X-Volume-Shape': '2,2,2' },
  }))
  await assert.rejects(generateVolume([[0]], 0, [1, 1, 1]), /does not match/)
})

test('preparation preserves fractional channels and forwards cancellation', async t => {
  const image = [[[0.25]], [[0.75]]]
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/prepare')
    assert.equal(options.signal, controller.signal)
    assert.deepEqual(JSON.parse(options.body), { image: [[0]] })
    return Response.json({ image })
  })
  assert.deepEqual(await prepareImage([[0]], controller.signal), image)
})

test('structured validation failures produce readable errors', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: [{ msg: 'invalid' }] }, { status: 422 }))
  await assert.rejects(prepareImage([[0]]), /HTTP 422/)
})
