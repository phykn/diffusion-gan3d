async function request(path, signal, payload) {
  const options = { signal }
  if (payload !== undefined) {
    options.method = 'POST'
    options.headers = { 'Content-Type': 'application/json' }
    options.body = JSON.stringify(payload)
  }
  const response = await fetch(path, options)
  if (response.ok) return response

  const { detail } = await response.json().catch(() => ({}))
  if (typeof detail === 'string') throw new Error(detail)
  if (Array.isArray(detail) && detail.length) {
    throw new Error(detail.map(item => item?.msg ?? String(item)).join('; '))
  }
  throw new Error([`HTTP ${response.status}`, response.statusText].filter(Boolean).join(' '))
}

export async function getHealth(signal) {
  const response = await request('/api/health', signal)
  return response.json()
}

export async function prepareImage(image, signal) {
  const response = await request('/api/prepare', signal, { image })
  const prepared = await response.json()
  return prepared.image
}

export async function generateVolume(options, signal) {
  const { image, axis, seed, blocks, heightOrigin, heightExtent, tortuosity } = options
  const payload = { anchors: [{ image, axis, index: 0, position: null }], seed, tortuosity }
  if (blocks.some(count => count > 1)) payload.blocks = blocks
  if (heightOrigin !== undefined) {
    payload.height_origin = heightOrigin
    payload.height_extent = heightExtent
  }
  const response = await request('/api/generate', signal, payload)
  const shape = (response.headers.get('X-Volume-Shape') ?? '').split(',').map(Number)
  const size = shape.reduce((product, count) => product * count, 1)
  if (shape.length !== 3 || shape.some(count => count < 1 || !Number.isSafeInteger(count)) || !Number.isSafeInteger(size)) {
    throw new Error('The API returned an invalid volume shape.')
  }
  const values = new Uint8Array(await response.arrayBuffer())
  if (values.length !== size) throw new Error('The returned volume size does not match its header.')

  const header = response.headers.get('X-Tortuosity')?.trim()
  const factor = header ? Number(header) : NaN
  return { shape, values, tortuosity: header === 'inf' ? Infinity : Number.isFinite(factor) ? factor : null }
}
