import assert from 'node:assert/strict'
import test from 'node:test'
import { createRenderer, nextTick, ref } from 'vue'
import { useGeneration } from '../src/use-generation.js'

const renderer = createRenderer({
  createComment: () => ({}), insert() {}, remove() {},
  parentNode: () => null, nextSibling: () => null,
})
const settle = () => new Promise(resolve => setImmediate(resolve))

async function mount(t, respond, health = {}) {
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    if (path === '/health') return Response.json({ crop_size: 1, num_phases: 2, ...health })
    if (path === '/prepare') {
      assert.deepEqual(JSON.parse(options.body).image, [[1]])
      return Response.json({ image: [[[0]], [[1]]] })
    }
    return respond(options)
  })
  let state
  const app = renderer.createApp({
    setup() {
      state = useGeneration(ref({ getAnchorImage: () => ({ image: [[1]], cropOrigin: [23, 4], sourceShape: [512, 100] }) }))
      return () => null
    },
  })
  app.mount({})
  t.after(() => app.unmount())
  await settle()
  state.selectFile({ name: 'section.png' })
  state.cropReady.value = true
  return { app, state }
}

function volume() {
  return new Response(new Uint8Array([1]), { headers: { 'X-Volume-Shape': '1,1,1' } })
}

test('publish one complete volume result and clear busy state', async t => {
  const { state } = await mount(t, async () => volume())
  await state.generate()
  assert.deepEqual(state.result.value.shape, [1, 1, 1])
  assert.deepEqual(state.result.value.values, new Uint8Array([1]))
  assert.equal(state.busy.value, false)
  assert.equal(state.error.value, false)
})

test('input changes discard a pending generation without leaving busy set', async t => {
  let finish
  const { state } = await mount(t, () => new Promise(resolve => { finish = resolve }))
  const pending = state.generate()
  await settle()
  state.seed.value += 1
  await nextTick()
  finish(volume())
  await pending
  assert.equal(state.result.value, null)
  assert.equal(state.busy.value, false)
})

test('unmount cancels pending network work and prevents publication', async t => {
  let signal
  const { state, app } = await mount(t, options => new Promise((resolve, reject) => {
    signal = options.signal
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
  }))
  const pending = state.generate()
  await settle()
  app.unmount()
  await pending
  assert.equal(signal.aborted, true)
  assert.equal(state.result.value, null)
  assert.equal(state.error.value, false)
})

test('height-conditioned sections send source geometry and the selected domain', async t => {
  let body
  const { state } = await mount(t, async options => {
    body = JSON.parse(options.body)
    return volume()
  }, { height_enabled: true, num_domains: 2, height_extents: { 0: null, 1: 768 } })
  assert.equal(state.plane.value, 1)
  state.domain.value = 1
  await nextTick()
  assert.equal(state.heightExtent.value, 768)
  await state.generate()
  assert.equal(body.anchors[0].axis, 1)
  assert.equal(body.height_origin, 23)
  assert.equal(body.height_extent, 512)
  assert.equal(body.domain, 1)
  state.plane.value = 0
  state.heightOrigin.value = 10
  await nextTick()
  await state.generate()
  assert.equal(body.anchors[0].axis, 0)
  assert.equal(body.height_origin, 10)
  assert.equal(body.height_extent, 768)
})

test('XY sections require an explicit Z extent when the model has mixed heights', async t => {
  const { state } = await mount(t, () => { throw new Error('must not generate') }, { height_enabled: true })
  state.plane.value = 0
  await nextTick()
  await state.generate()
  assert.equal(state.error.value, true)
  assert.match(state.status.value, /full Z extent/)
})
