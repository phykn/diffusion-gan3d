import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { generateVolume, getHealth, prepareImage } from './api.js'
import { createRequestGate, normalizeBlocks, normalizeSeed } from './input-state.js'

const HEALTH_RETRY_MS = 500
const HEALTH_RETRY_MAX_MS = 5000

function wait(delay, signal) {
  return new Promise(resolve => {
    const done = () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', done)
      resolve()
    }
    const timer = setTimeout(done, delay)
    signal.addEventListener('abort', done, { once: true })
  })
}

function phaseFraction(values, phase) {
  let count = 0
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] === phase) count += 1
  }
  return values.length ? count / values.length : null
}

export function useGeneration(cropEditor) {
  const cropReady = ref(false)
  const file = ref(null)
  const seed = ref(0)
  const blocks = ref([1, 1, 1])
  const axis = ref(0)
  const domain = ref(0)
  const heightOrigin = ref(0)
  const heightExtent = ref(null)
  const busy = ref(false)
  const error = ref('')
  const health = ref(null)
  const healthError = ref('')
  const measureTortuosity = ref(true)
  const result = ref(null)
  const ready = computed(() => Boolean(
    file.value && cropReady.value && health.value && !busy.value,
  ))
  const gate = createRequestGate()
  const controller = new AbortController()

  function heightExtentFor(value) {
    return health.value?.height_extents?.[value] ?? null
  }

  // The backend may start after the page; keep retrying until it answers.
  onMounted(async () => {
    let delay = HEALTH_RETRY_MS
    while (!controller.signal.aborted) {
      try {
        health.value = await getHealth(controller.signal)
        healthError.value = ''
        axis.value = health.value.height_enabled ? 1 : 0
        heightExtent.value = heightExtentFor(domain.value)
        return
      } catch {
        if (controller.signal.aborted) return
        healthError.value = 'Inference server unreachable · retrying'
        await wait(delay, controller.signal)
        delay = Math.min(delay * 2, HEALTH_RETRY_MAX_MS)
      }
    }
  })
  onBeforeUnmount(() => {
    gate.invalidate()
    controller.abort()
  })

  function invalidateResult() {
    gate.invalidate()
    result.value = null
    error.value = ''
  }

  function selectFile(nextFile) {
    invalidateResult()
    file.value = nextFile
    cropReady.value = false
  }

  watch([seed, axis, domain, heightOrigin, heightExtent], invalidateResult)
  watch(blocks, invalidateResult, { deep: true })
  watch(domain, value => {
    heightExtent.value = heightExtentFor(value)
  })

  async function generate() {
    if (!ready.value) return
    const token = gate.begin()
    busy.value = true
    error.value = ''
    try {
      const crop = cropEditor.value?.getAnchorImage()
      if (!crop) {
        throw new Error(`Choose a valid ${health.value.crop_size} × ${health.value.crop_size} crop first.`)
      }
      const options = {
        axis: axis.value,
        seed: normalizeSeed(seed.value),
        blocks: normalizeBlocks(blocks.value, health.value.max_axis_blocks),
        domain: domain.value,
        tortuosity: measureTortuosity.value,
      }
      if (health.value.height_enabled) {
        options.heightOrigin = axis.value === 0 ? heightOrigin.value : crop.cropOrigin[0]
        options.heightExtent = axis.value === 0 ? heightExtent.value : crop.sourceShape[0]
        if (
          !Number.isFinite(options.heightOrigin) || options.heightOrigin < 0
          || !Number.isFinite(options.heightExtent) || options.heightExtent <= 0
        ) {
          throw new Error('Enter a non-negative Z origin and a positive full Z extent in source pixels.')
        }
      }
      const image = await prepareImage(crop.image, controller.signal)
      if (!gate.accepts(token)) return
      const next = await generateVolume({ ...options, image }, controller.signal)
      if (gate.accepts(token)) {
        result.value = { ...next, porosity: phaseFraction(next.values, health.value.pore_phase) }
      }
    } catch (reason) {
      if (gate.accepts(token)) error.value = reason instanceof Error ? reason.message : String(reason)
    } finally {
      busy.value = false
    }
  }

  return {
    cropReady,
    file,
    seed,
    blocks,
    axis,
    domain,
    heightOrigin,
    heightExtent,
    busy,
    error,
    health,
    healthError,
    measureTortuosity,
    result,
    ready,
    invalidateResult,
    selectFile,
    generate,
  }
}
