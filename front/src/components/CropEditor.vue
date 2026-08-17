<script setup>
import { nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

const props = defineProps({
  file: File,
  cropSize: { type: Number, default: 128 },
  inputSize: { type: Number, default: 128 },
  disabled: Boolean,
})
const emit = defineEmits(['change', 'error', 'ready'])
const canvas = ref(null)
const state = reactive({ image: null, crop: null, view: null, drag: null })
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
  const size = state.crop.size * scale
  context.fillStyle = 'rgb(20 20 19 / 48%)'
  context.beginPath()
  context.rect(x, y, drawWidth, drawHeight)
  context.rect(cx, cy, size, size)
  context.fill('evenodd')
  context.strokeStyle = '#cc785c'
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

function clamp() {
  state.crop.x = Math.max(0, Math.min(state.crop.x, state.image.naturalWidth - state.crop.size))
  state.crop.y = Math.max(0, Math.min(state.crop.y, state.image.naturalHeight - state.crop.size))
}

function pointerDown(event) {
  if (props.disabled || !state.crop) return
  const next = point(event)
  const inside = next.x >= state.crop.x
    && next.x <= state.crop.x + state.crop.size
    && next.y >= state.crop.y
    && next.y <= state.crop.y + state.crop.size
  if (!inside) {
    const previousX = state.crop.x
    const previousY = state.crop.y
    state.crop.x = next.x - state.crop.size / 2
    state.crop.y = next.y - state.crop.size / 2
    clamp()
    if (state.crop.x !== previousX || state.crop.y !== previousY) emit('change')
  }
  state.drag = { x: next.x - state.crop.x, y: next.y - state.crop.y }
  canvas.value.setPointerCapture(event.pointerId)
  draw()
}

function pointerMove(event) {
  if (props.disabled || !state.drag) return
  const next = point(event)
  const previousX = state.crop.x
  const previousY = state.crop.y
  state.crop.x = next.x - state.drag.x
  state.crop.y = next.y - state.drag.y
  clamp()
  if (state.crop.x !== previousX || state.crop.y !== previousY) emit('change')
  draw()
}

watch(() => props.disabled, (disabled) => {
  if (disabled) state.drag = null
})

watch(() => [props.file, props.cropSize], ([file]) => {
  const version = ++loadVersion
  state.image = null
  state.crop = null
  emit('ready', false)
  draw()
  if (!file) {
    return
  }
  const cropSize = props.cropSize
  const url = URL.createObjectURL(file)
  const image = new Image()
  image.onload = async () => {
    URL.revokeObjectURL(url)
    if (version !== loadVersion) return
    if (image.naturalWidth < cropSize || image.naturalHeight < cropSize) {
      emit('error', `Image must be at least ${cropSize} × ${cropSize}.`)
      return
    }
    state.image = image
    state.crop = {
      x: (image.naturalWidth - cropSize) / 2,
      y: (image.naturalHeight - cropSize) / 2,
      size: cropSize,
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
  if (!state.image || !state.crop) return null
  const output = document.createElement('canvas')
  output.width = props.inputSize
  output.height = props.inputSize
  const context = output.getContext('2d', { willReadFrequently: true })
  context.imageSmoothingEnabled = false
  context.drawImage(state.image, state.crop.x, state.crop.y, props.cropSize, props.cropSize, 0, 0, props.inputSize, props.inputSize)
  const pixels = context.getImageData(0, 0, props.inputSize, props.inputSize).data
  const values = new Float32Array(props.inputSize * props.inputSize)
  let low = Infinity
  let high = -Infinity
  for (let i = 0; i < values.length; i += 1) {
    const j = i * 4
    const value = .2126 * pixels[j] + .7152 * pixels[j + 1] + .0722 * pixels[j + 2]
    values[i] = value
    low = Math.min(low, value)
    high = Math.max(high, value)
  }
  if (high - low < .5) throw new Error('The selected crop contains only one visible phase.')
  const threshold = (low + high) / 2
  return Array.from({ length: props.inputSize }, (_, row) => Array.from({ length: props.inputSize }, (_, column) => values[row * props.inputSize + column] > threshold ? 1 : 0))
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
    <header class="panel-header"><div><h2>Image</h2></div></header>
    <div class="canvas-wrap crop-canvas" :class="{ 'crop-canvas--disabled': disabled }"><canvas ref="canvas" @pointerdown="pointerDown" @pointermove="pointerMove" @pointerup="state.drag = null" @pointercancel="state.drag = null"/><div v-if="!state.image" class="empty-state"><strong>No section selected</strong></div></div>
  </section>
</template>
