import assert from 'node:assert/strict'
import { createServer as createHttpServer } from 'node:http'
import { once } from 'node:events'
import test from 'node:test'
import { createServer } from 'vite'
import config from '../vite.config.js'

test('development server forwards preparation, health and generation', async t => {
  const backend = createHttpServer((request, response) => {
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify({ path: request.url, method: request.method }))
  })
  backend.listen(0, '127.0.0.1')
  await once(backend, 'listening')
  t.after(() => new Promise(resolve => backend.close(resolve)))
  const target = `http://127.0.0.1:${backend.address().port}`
  const proxy = Object.fromEntries(Object.keys(config.server.proxy).map(path => [path, target]))
  const server = await createServer({
    configFile: false,
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { host: '127.0.0.1', port: 0, proxy },
  })
  t.after(() => server.close())
  await server.listen()
  const origin = `http://127.0.0.1:${server.httpServer.address().port}`
  for (const [path, method] of [['/health', 'GET'], ['/prepare', 'POST'], ['/generate', 'POST']]) {
    const response = await fetch(`${origin}${path}`, { method })
    assert.equal(response.status, 200)
    assert.deepEqual(await response.json(), { path, method })
  }
})
