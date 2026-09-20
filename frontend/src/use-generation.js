import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { generateVolume, getHealth, prepareImage } from './api.js'
import { createRequestGate, normalizeBlocks, normalizeSeed } from './input-state.js'

export function useGeneration(cropEditor) {
  const cropReady = ref(false)
  const file = ref(null)
  const seed = ref(0)
  const blocks = ref([1, 1, 1])
  const busy = ref(false)
  const status = ref('')
  const error = ref(false)
  const health = ref(null)
  const result = ref(null)
  const ready = computed(() => Boolean(file.value && cropReady.value && health.value && !busy.value))
  const gate = createRequestGate()
  const controller = new AbortController()

  onMounted(async () => {
    try {
      health.value = await getHealth(controller.signal)
    } catch {
      if (controller.signal.aborted) return
      error.value = true
      status.value = 'Could not connect to the inference server.'
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

  async function generate() {
    if (!ready.value) return
    const token = gate.begin()
    busy.value = true
    error.value = false
    status.value = ''
    try {
      const crop = cropEditor.value?.getAnchorImage()
      if (!crop) throw new Error(`Choose a valid ${health.value.crop_size} × ${health.value.crop_size} crop first.`)
      const requestSeed = normalizeSeed(seed.value)
      const requestBlocks = normalizeBlocks(blocks.value)
      const image = await prepareImage(crop, controller.signal)
      if (!gate.accepts(token)) return
      const next = await generateVolume(image, requestSeed, requestBlocks, controller.signal)
      if (gate.accepts(token)) result.value = next
    } catch (reason) {
      if (!gate.accepts(token)) return
      error.value = true
      status.value = reason instanceof Error ? reason.message : String(reason)
    } finally {
      if (gate.ownsRequest(token)) busy.value = false
    }
  }

  return { cropReady, file, seed, blocks, busy, status, error, health, result, ready, invalidateResult, selectFile, generate }
}
