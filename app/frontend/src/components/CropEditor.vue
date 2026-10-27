<script setup>
import { nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { clampCrop, cropAnchor, readPngLabels } from '../image-labels.js'

const props = defineProps({
  file: File,
  cropSize: Number,
  disabled: Boolean,
})
const emit = defineEmits(['change', 'error', 'ready'])
const canvas = ref(null)
const state = reactive({ image: null, sourceLabels: null, crop: null, view: null, drag: null })
let observer
let loadVersion = 0

function fitCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const rect = canvas.value.getBoundingClientRect()
  const width = Math.max(1, Math.round(rect.width * dpr))
  const height = Math.max(1, Math.round(rect.height * dpr))
  if (canvas.value.width !== width || canvas.value.height !== height) {
    canvas.value.width = width
    canvas.value.height = height
  }
  return { width, height, dpr }
}

function draw() {
  if (!canvas.value) return
  const { width, height, dpr } = fitCanvas()
  const context = canvas.value.getContext('2d')
  context.clearRect(0, 0, width, height)
  if (!state.image || !state.crop) return
  const padding = 24 * dpr
  const scale = Math.min(
    (width - 2 * padding) / state.image.naturalWidth,
    (height - 2 * padding) / state.image.naturalHeight,
  )
  const drawWidth = state.image.naturalWidth * scale
  const drawHeight = state.image.naturalHeight * scale
  const x = (width - drawWidth) / 2
  const y = (height - drawHeight) / 2
  state.view = { x, y, scale }
  context.drawImage(state.image, x, y, drawWidth, drawHeight)

  const cx = x + state.crop.x * scale
  const cy = y + state.crop.y * scale
  const size = props.cropSize * scale
  context.fillStyle = 'rgb(20 20 19 / 48%)'
  context.beginPath()
  context.rect(x, y, drawWidth, drawHeight)
  context.rect(cx, cy, size, size)
  context.fill('evenodd')
  context.strokeStyle = getComputedStyle(canvas.value).getPropertyValue('--primary')
  context.lineWidth = 2 * dpr
  context.strokeRect(cx, cy, size, size)
  context.fillStyle = '#fff'
  context.font = `${11 * dpr}px system-ui`
  context.fillText(`${props.cropSize} × ${props.cropSize}`, cx + 9 * dpr, cy + 18 * dpr)
}

function point(event) {
  const rect = canvas.value.getBoundingClientRect()
  return {
    x: (event.offsetX * canvas.value.width / rect.width - state.view.x) / state.view.scale,
    y: (event.offsetY * canvas.value.height / rect.height - state.view.y) / state.view.scale,
  }
}

function moveCrop(x, y) {
  const next = clampCrop(state.sourceLabels, { x, y }, props.cropSize)
  if (next.x === state.crop.x && next.y === state.crop.y) return
  Object.assign(state.crop, next)
  emit('change')
}

function pointerDown(event) {
  if (props.disabled || !state.crop) return
  const next = point(event)
  const size = props.cropSize
  const inside = next.x >= state.crop.x
    && next.x <= state.crop.x + size
    && next.y >= state.crop.y
    && next.y <= state.crop.y + size
  if (!inside) moveCrop(next.x - size / 2, next.y - size / 2)
  state.drag = { x: next.x - state.crop.x, y: next.y - state.crop.y }
  canvas.value.setPointerCapture(event.pointerId)
  draw()
}

function pointerMove(event) {
  if (props.disabled || !state.drag) return
  const next = point(event)
  moveCrop(next.x - state.drag.x, next.y - state.drag.y)
  draw()
}

watch(() => props.disabled, (disabled) => {
  if (disabled) state.drag = null
})

watch(() => [props.file, props.cropSize], ([file], previous, onCleanup) => {
  const version = ++loadVersion
  state.image = null
  state.sourceLabels = null
  state.crop = null
  emit('ready', false)
  draw()
  // Wait for /health: a guessed size could reject an image the model accepts.
  if (!file || !props.cropSize) return
  const cropSize = props.cropSize
  const url = URL.createObjectURL(file)
  onCleanup(() => URL.revokeObjectURL(url))
  const image = new Image()
  image.onload = async () => {
    URL.revokeObjectURL(url)
    if (version !== loadVersion) return
    if (image.naturalWidth < cropSize || image.naturalHeight < cropSize) {
      emit('error', `Image must be at least ${cropSize} × ${cropSize}.`)
      return
    }
    let sourceLabels = null
    try {
      sourceLabels = await readPngLabels(file)
    } catch (reason) {
      if (version === loadVersion) emit('error', reason instanceof Error ? reason.message : 'Could not read PNG labels.')
      return
    }
    if (version !== loadVersion) return
    state.image = image
    state.sourceLabels = sourceLabels
    state.crop = {
      x: Math.floor((sourceLabels.width - cropSize) / 2),
      y: Math.floor((sourceLabels.height - cropSize) / 2),
    }
    emit('ready', true)
    await nextTick()
    draw()
  }
  image.onerror = () => {
    URL.revokeObjectURL(url)
    if (version === loadVersion) emit('error', 'The browser could not decode this image.')
  }
  image.src = url
})

function getAnchorImage() {
  return state.crop && cropAnchor(state.sourceLabels, state.crop, props.cropSize)
}

onMounted(() => {
  observer = new ResizeObserver(draw)
  observer.observe(canvas.value)
})
onBeforeUnmount(() => {
  loadVersion += 1
  observer?.disconnect()
})
defineExpose({ getAnchorImage })
</script>

<template>
  <section class="panel">
    <header class="panel-header">
      <div><h2>Image</h2></div>
    </header>
    <div class="canvas-wrap" :class="{ 'crop-canvas--disabled': disabled }">
      <canvas
        ref="canvas"
        @pointerdown="pointerDown"
        @pointermove="pointerMove"
        @pointerup="state.drag = null"
        @pointercancel="state.drag = null"
      />
      <div v-if="!state.image" class="empty-state">
        <strong>No section selected</strong>
      </div>
    </div>
  </section>
</template>
