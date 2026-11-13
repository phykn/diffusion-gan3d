import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { generateVolume, getHealth, prepareImage } from './api.js'
import { normalizeBlocks, normalizeSeed } from './input-state.js'

function retryAfter(delay, signal) {
  if (signal.aborted) return Promise.resolve()
  return new Promise(resolve => {
    const finish = () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', finish)
      resolve()
    }
    const timer = setTimeout(finish, delay)
    signal.addEventListener('abort', finish, { once: true })
  })
}

export function useGeneration(cropEditor) {
  const file = ref(null)
  const cropReady = ref(false)
  const health = ref(null)
  const healthError = ref('')
  const seed = ref(0)
  const blocks = ref([1, 1, 1])
  const axis = ref(0)
  const heightOrigin = ref(0)
  const heightExtent = ref(null)
  const measureTortuosity = ref(true)
  const busy = ref(false)
  const result = ref(null)
  const error = ref('')
  const ready = computed(() => !!(file.value && cropReady.value && health.value && !busy.value))
  const lifetime = new AbortController()
  let revision = 0

  async function connect() {
    const { signal } = lifetime
    for (let delay = 500; !signal.aborted; delay = Math.min(delay * 2, 5000)) {
      try {
        const model = await getHealth(signal)
        if (signal.aborted) return
        health.value = model
        axis.value = model.height_enabled ? 1 : 0
        heightExtent.value = model.height_extent ?? null
        healthError.value = ''
        return
      } catch (reason) {
        if (signal.aborted) return
        healthError.value = reason instanceof TypeError
          ? 'Inference server unreachable · retrying'
          : `Inference server error: ${reason.message} · retrying`
        await retryAfter(delay, signal)
      }
    }
  }

  function invalidateResult() {
    revision += 1
    result.value = null
    error.value = ''
  }

  function selectFile(selected) {
    invalidateResult()
    cropReady.value = false
    file.value = selected
  }

  function optionsFor(crop) {
    const model = health.value
    const options = {
      axis: axis.value,
      seed: normalizeSeed(seed.value),
      blocks: normalizeBlocks(blocks.value, model.max_axis_blocks),
      tortuosity: measureTortuosity.value,
    }
    if (model.height_enabled) {
      const [origin, extent] = axis.value === 0
        ? [heightOrigin.value, heightExtent.value]
        : [crop.cropOrigin[0], crop.sourceShape[0]]
      if (!(Number.isFinite(origin) && origin >= 0 && Number.isFinite(extent) && extent > 0)) {
        throw new Error('Enter a non-negative Z origin and a positive full Z extent in source pixels.')
      }
      Object.assign(options, { heightOrigin: origin, heightExtent: extent })
    }
    return options
  }

  async function generate() {
    if (!ready.value) return
    const { signal } = lifetime
    const inputRevision = revision
    const current = () => !signal.aborted && revision === inputRevision
    busy.value = true
    error.value = ''
    try {
      const crop = cropEditor.value?.getAnchorImage()
      if (!crop) {
        throw new Error(`Choose a valid ${health.value.crop_size} × ${health.value.crop_size} crop first.`)
      }
      const options = optionsFor(crop)
      const image = await prepareImage(crop.image, signal)
      if (!current()) return
      const volume = await generateVolume({ ...options, image }, signal)
      if (!current()) return
      let pores = 0
      for (const label of volume.values) if (label === health.value.pore_phase) pores++
      result.value = { ...volume, porosity: pores / volume.values.length }
    } catch (reason) {
      if (current()) error.value = reason instanceof Error ? reason.message : String(reason)
    } finally {
      busy.value = false
    }
  }

  watch([seed, axis, heightOrigin, heightExtent, blocks], invalidateResult, {
    deep: true,
    flush: 'sync',
  })
  onMounted(connect)
  onBeforeUnmount(() => lifetime.abort())

  return {
    file, cropReady, health, healthError, seed, blocks, axis, heightOrigin, heightExtent,
    measureTortuosity, busy, result, error, ready, selectFile, invalidateResult, generate,
  }
}
