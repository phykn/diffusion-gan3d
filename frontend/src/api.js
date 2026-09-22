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
    const detail = typeof problem.detail === 'string' ? problem.detail : `HTTP ${response.status}`
    throw new Error(detail)
  }
  return response
}

export async function getHealth(signal) {
  return (await request('/health', undefined, signal)).json()
}

export async function prepareImage(image, signal) {
  const response = await request('/prepare', { image }, signal)
  return (await response.json()).image
}

function readMetric(headers, name) {
  const value = headers.get(name)
  if (value === null || value.trim() === '' || value === 'unavailable') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export async function generateVolume(image, seed, blocks, signal, conditions = {}) {
  const { axis = 0, domain = 0, height_origin, height_extent } = conditions
  const response = await request('/generate', {
    anchors: [{ image, axis, index: 0, position: [0, 0] }],
    domain,
    ...(height_origin === undefined ? {} : { height_origin, height_extent }),
    seed,
    ...(blocks.some(value => value > 1) ? { blocks } : {}),
    format: 'raw',
    include_metrics: true,
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
  return {
    shape,
    values,
    porosity: readMetric(response.headers, 'X-Porosity'),
    tortuosity: readMetric(response.headers, 'X-Tortuosity'),
  }
}
