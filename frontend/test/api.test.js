import assert from 'node:assert/strict'
import test from 'node:test'
import { generateVolume, prepareImage } from '../src/api.js'

test('generation sends a prepared boundary anchor and decodes volume metadata', async t => {
  const image = [[[0.25]], [[0.75]]]
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/api/generate')
    assert.deepEqual(JSON.parse(options.body), {
      anchors: [{ image, axis: 0, index: 0, position: [0, 0] }],
      domain: 0,
      seed: 7,
      blocks: [2, 1, 1],
      format: 'raw',
      tortuosity: true,
    })
    return new Response(new Uint8Array([0, 1]), { headers: {
      'X-Volume-Shape': '2,1,1', 'X-Tortuosity': 'unavailable',
    } })
  })
  const result = await generateVolume(image, 7, [2, 1, 1])
  assert.deepEqual(result, { shape: [2, 1, 1], values: new Uint8Array([0, 1]), porosity: 0.5, tortuosity: null })
})

test('porosity is counted locally when the server skips tortuosity', async t => {
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(JSON.parse(options.body).tortuosity, false)
    return new Response(new Uint8Array([0, 0, 0, 1]), { headers: { 'X-Volume-Shape': '4,1,1' } })
  })
  const result = await generateVolume([[0]], 0, [1, 1, 1], undefined, {}, { tortuosity: false })
  assert.equal(result.porosity, 0.75)
  assert.equal(result.tortuosity, null)
})

test('a phase without a spanning path reports infinite tortuosity', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([1]), {
    headers: { 'X-Volume-Shape': '1,1,1', 'X-Tortuosity': 'inf' },
  }))
  assert.equal((await generateVolume([[0]], 0, [1, 1, 1])).tortuosity, Infinity)
})

test('single-block requests omit tiled geometry', async t => {
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal('blocks' in JSON.parse(options.body), false)
    return new Response(new Uint8Array([0]), { headers: { 'X-Volume-Shape': '1,1,1' } })
  })
  await generateVolume([[0]], 0, [1, 1, 1])
})

test('side-section request preserves the physical crop origin in a tiled volume', async t => {
  const image = [[[0.25]], [[0.75]]]
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    const body = JSON.parse(options.body)
    assert.deepEqual(body.anchors, [{ image, axis: 2, index: 0, position: [0, 0] }])
    assert.equal(body.domain, 1)
    assert.equal(body.height_origin, 23)
    assert.equal(body.height_extent, 512)
    return new Response(new Uint8Array([0]), { headers: { 'X-Volume-Shape': '1,1,1' } })
  })
  await generateVolume(image, 0, [2, 2, 2], undefined, { axis: 2, domain: 1, height_origin: 23, height_extent: 512 })
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
    assert.equal(path, '/api/prepare')
    assert.equal(options.signal, controller.signal)
    assert.deepEqual(JSON.parse(options.body), { image: [[0]] })
    return Response.json({ image })
  })
  assert.deepEqual(await prepareImage([[0]], controller.signal), image)
})

test('structured validation failures produce readable errors', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json(
    { detail: [{ msg: 'value too large' }, { msg: 'missing field' }] }, { status: 422 },
  ))
  await assert.rejects(prepareImage([[0]]), /value too large; missing field/)
})

test('plain-text server errors keep their status text', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Internal Server Error', {
    status: 500, statusText: 'Internal Server Error',
  }))
  await assert.rejects(prepareImage([[0]]), /HTTP 500 Internal Server Error/)
})
