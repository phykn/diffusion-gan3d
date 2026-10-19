import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { generateVolume, getHealth, prepareImage } from './api.js'
import { createRequestGate, normalizeBlocks, normalizeSeed } from './input-state.js'

const HEALTH_RETRY_MS = 500
const HEALTH_RETRY_MAX_MS = 5000

function wait(delay, signal) {
  return new Promise(resolve => {
    const timer = setTimeout(resolve, delay)
    signal.addEventListener('abort', () => {
      clearTimeout(timer)
      resolve()
    }, { once: true })
  })
}

export function useGeneration(cropEditor) {
  const cropReady = ref(false)
  const file = ref(null)
  const seed = ref(0)
  const blocks = ref([1, 1, 1])
  const plane = ref(0)
  const domain = ref(0)
  const heightOrigin = ref(0)
  const heightExtent = ref(null)
  const busy = ref(false)
  const status = ref('')
  const error = ref(false)
  const health = ref(null)
  const healthError = ref('')
  const measureTortuosity = ref(true)
  const result = ref(null)
  const ready = computed(() => Boolean(
    file.value && cropReady.value && health.value && !busy.value,
  ))
  const gate = createRequestGate()
  const controller = new AbortController()

  // The backend may start after the page; keep retrying until it answers.
  onMounted(async () => {
    for (let delay = HEALTH_RETRY_MS; !controller.signal.aborted; delay = Math.min(delay * 2, HEALTH_RETRY_MAX_MS)) {
      try {
        health.value = await getHealth(controller.signal)
        healthError.value = ''
        plane.value = health.value.height_enabled ? 1 : 0
        heightExtent.value = health.value.height_extents?.[domain.value] ?? null
        return
      } catch {
        if (controller.signal.aborted) return
        healthError.value = 'Inference server unreachable · retrying'
        await wait(delay, controller.signal)
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
    error.value = false
    status.value = ''
  }

  function selectFile(nextFile) {
    invalidateResult()
    file.value = nextFile
    cropReady.value = false
  }

  watch(seed, invalidateResult)
  watch(blocks, invalidateResult, { deep: true })
  watch([plane, domain, heightOrigin, heightExtent], invalidateResult)
  watch(domain, value => {
    heightExtent.value = health.value?.height_extents?.[value] ?? null
  })

  async function generate() {
    if (!ready.value) return
    const token = gate.begin()
    busy.value = true
    error.value = false
    status.value = ''
    try {
      const crop = cropEditor.value?.getAnchorImage()
      if (!crop) {
        throw new Error(`Choose a valid ${health.value.crop_size} × ${health.value.crop_size} crop first.`)
      }
      const requestSeed = normalizeSeed(seed.value)
      const requestBlocks = normalizeBlocks(blocks.value, health.value.max_blocks)
      const conditions = { axis: plane.value, domain: domain.value }
      if (health.value.height_enabled) {
        conditions.height_origin = plane.value === 0 ? heightOrigin.value : crop.cropOrigin[0]
        conditions.height_extent = plane.value === 0 ? heightExtent.value : crop.sourceShape[0]
        if (
          !Number.isFinite(conditions.height_origin) || conditions.height_origin < 0
          || !Number.isFinite(conditions.height_extent) || conditions.height_extent <= 0
        ) {
          throw new Error('Enter a non-negative Z origin and a positive full Z extent in source pixels.')
        }
      }
      const image = await prepareImage(crop.image, controller.signal)
      if (!gate.accepts(token)) return
      const next = await generateVolume(
        image, requestSeed, requestBlocks, controller.signal, conditions,
        { tortuosity: measureTortuosity.value },
      )
      if (gate.accepts(token)) result.value = next
    } catch (reason) {
      if (!gate.accepts(token)) return
      error.value = true
      status.value = reason instanceof Error ? reason.message : String(reason)
    } finally {
      if (gate.ownsRequest(token)) busy.value = false
    }
  }

  return {
    cropReady,
    file,
    seed,
    blocks,
    plane,
    domain,
    heightOrigin,
    heightExtent,
    busy,
    status,
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
