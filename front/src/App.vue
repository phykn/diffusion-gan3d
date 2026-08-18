<script setup>
import { computed, defineAsyncComponent, onMounted, ref, watch } from 'vue'
import CropEditor from './components/CropEditor.vue'
import Sidebar from './components/Sidebar.vue'
import { createRequestGate, normalizeBlocks, normalizeSeed } from './input-state.js'

const VolumeViewer = defineAsyncComponent(() => import('./components/VolumeViewer.vue'))

const cropEditor = ref(null)
const cropReady = ref(false)
const file = ref(null)
const seed = ref(0)
const blocks = ref([1, 1, 1])
const busy = ref(false)
const status = ref('')
const error = ref(false)
const health = ref(null)
const volume = ref(null)
const shape = ref(null)
const porosity = ref(null)
const tortuosity = ref(null)
const ready = computed(() => Boolean(file.value && cropReady.value && health.value && !busy.value))
const requestGate = createRequestGate()

onMounted(async () => {
  try {
    const response = await fetch('/health')
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    health.value = await response.json()
  } catch {
    error.value = true
    status.value = 'Could not connect to the inference server.'
  }
})

function invalidateResult() {
  requestGate.invalidate()
  volume.value = null
  shape.value = null
  porosity.value = null
  tortuosity.value = null
  error.value = false
}

function selectFile(nextFile) {
  invalidateResult()
  file.value = nextFile
  cropReady.value = false
}

watch(seed, invalidateResult)
watch(blocks, invalidateResult, { deep: true })

async function generate() {
  const token = requestGate.begin()
  try {
    const image = cropEditor.value?.getAnchorImage()
    if (!image) throw new Error(`Choose a valid ${health.value?.crop_size ?? 128} × ${health.value?.crop_size ?? 128} crop first.`)
    const requestSeed = normalizeSeed(seed.value)
    const requestBlocks = normalizeBlocks(blocks.value)
    busy.value = true
    error.value = false
    const response = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anchors: [{ image, axis: 0, index: 0 }],
        seed: requestSeed,
        ...(requestBlocks.some(value => value > 1) ? { blocks: requestBlocks } : {}),
        format: 'raw',
      }),
    })
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}))
      throw new Error(problem.detail || `Generation failed with HTTP ${response.status}.`)
    }
    const nextShape = response.headers.get('X-Volume-Shape')?.split(',').map(Number)
    if (!nextShape || nextShape.length !== 3) throw new Error('The API returned an invalid volume shape.')
    const values = new Uint8Array(await response.arrayBuffer())
    if (values.length !== nextShape.reduce((total, value) => total * value, 1)) throw new Error('The returned volume size does not match its header.')
    if (!requestGate.accepts(token)) return
    shape.value = nextShape
    volume.value = values
    const rawPorosity = response.headers.get('X-Porosity')
    porosity.value = rawPorosity === null ? null : Number(rawPorosity)
    const tau = response.headers.get('X-Tortuosity')
    tortuosity.value = tau && tau !== 'unavailable' ? Number(tau) : null
  } catch (reason) {
    if (!requestGate.accepts(token)) return
    error.value = true
    status.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    if (requestGate.ownsRequest(token)) busy.value = false
  }
}
</script>

<template>
  <main class="app">
    <Sidebar v-model:seed="seed" v-model:blocks="blocks" :file :health :busy :ready :status :error :porosity :tortuosity @select-file="selectFile" @generate="generate" />
    <section class="editor-main">
      <div class="workspace">
        <CropEditor
          ref="cropEditor"
          :file
          :crop-size="health?.crop_size ?? 128"
          :input-size="health?.input_size ?? 128"
          :num-phases="health?.num_phases ?? 2"
          :disabled="busy"
          @change="invalidateResult"
          @error="status = $event; error = true"
          @ready="cropReady = $event"
        />
        <VolumeViewer :values="volume" :shape :busy :num-phases="health?.num_phases ?? 2" />
      </div>
    </section>
  </main>
</template>
