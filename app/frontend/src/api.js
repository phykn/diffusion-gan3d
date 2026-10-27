async function request(path, body, signal) {
  const response = await fetch(path, {
    signal,
    ...(body === undefined ? {} : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  })
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}))
    throw new Error(errorDetail(response, problem.detail))
  }
  return response
}

function errorDetail(response, detail) {
  if (typeof detail === 'string') return detail
  // FastAPI reports request validation failures as a list of {loc, msg}.
  if (Array.isArray(detail) && detail.length) {
    return detail.map(item => item?.msg ?? String(item)).join('; ')
  }
  return `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}`
}

export async function getHealth(signal) {
  return (await request('/api/health', undefined, signal)).json()
}

export async function prepareImage(image, signal) {
  const response = await request('/api/prepare', { image }, signal)
  return (await response.json()).image
}

function parseTortuosity(value) {
  if (value === null || value.trim() === '' || value === 'unavailable') return null
  // A phase without a spanning path has infinite tortuosity.
  if (value === 'inf') return Infinity
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export async function generateVolume(
  { image, axis, seed, blocks, domain, heightOrigin, heightExtent, tortuosity }, signal,
) {
  const response = await request('/api/generate', {
    anchors: [{ image, axis, index: 0, position: [0, 0] }],
    domain,
    ...(heightOrigin === undefined ? {} : { height_origin: heightOrigin, height_extent: heightExtent }),
    seed,
    ...(blocks.some(value => value > 1) ? { blocks } : {}),
    format: 'raw',
    // The server solves tortuosity before streaming.
    tortuosity,
  }, signal)
  const shape = response.headers.get('X-Volume-Shape')?.split(',').map(Number)
  const size = shape?.reduce((total, value) => total * value, 1)
  if (
    !shape || shape.length !== 3
    || !shape.every(value => Number.isSafeInteger(value) && value > 0)
    || !Number.isSafeInteger(size)
  ) {
    throw new Error('The API returned an invalid volume shape.')
  }
  const values = new Uint8Array(await response.arrayBuffer())
  if (values.length !== size) throw new Error('The returned volume size does not match its header.')
  return { shape, values, tortuosity: parseTortuosity(response.headers.get('X-Tortuosity')) }
}
