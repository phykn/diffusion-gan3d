<script setup>
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { clampCrop, cropAnchor, readPngLabels } from '../image-labels.js'

const props = defineProps({ file: File, cropSize: Number, disabled: Boolean })
const emit = defineEmits(['change', 'error', 'ready'])
const canvas = ref(null)
const preview = shallowRef(null)
let drag, view, observer

function paint() {
  const element = canvas.value
  if (!element) return
  const { width: cssWidth, height: cssHeight } = element.getBoundingClientRect()
  const ratio = Math.min(window.devicePixelRatio || 1, 2)
  const width = Math.max(1, Math.round(cssWidth * ratio))
  const height = Math.max(1, Math.round(cssHeight * ratio))
  if (element.width !== width) element.width = width
  if (element.height !== height) element.height = height
  const context = element.getContext('2d')
  context.clearRect(0, 0, width, height)
  view = null
  if (!preview.value || width <= 48 * ratio || height <= 48 * ratio) return

  const { image, source, crop } = preview.value
  const scale = Math.min((width - 48 * ratio) / source.width, (height - 48 * ratio) / source.height)
  const imageWidth = source.width * scale
  const imageHeight = source.height * scale
  const x = (width - imageWidth) / 2
  const y = (height - imageHeight) / 2
  view = { x, y, scale }
  context.drawImage(image, x, y, imageWidth, imageHeight)
  const left = x + crop.x * scale
  const top = y + crop.y * scale
  const side = props.cropSize * scale
  context.beginPath()
  context.rect(x, y, imageWidth, imageHeight)
  context.rect(left, top, side, side)
  context.fillStyle = 'rgb(20 20 19 / 48%)'
  context.fill('evenodd')
  context.strokeStyle = getComputedStyle(element).getPropertyValue('--primary')
  context.lineWidth = 2 * ratio
  context.strokeRect(left, top, side, side)
  context.fillStyle = '#fff'
  context.font = `${11 * ratio}px system-ui`
  context.fillText(`${props.cropSize} × ${props.cropSize}`, left + 9 * ratio, top + 18 * ratio)
}

function imagePoint(event) {
  const element = canvas.value
  const box = element.getBoundingClientRect()
  return {
    x: ((event.clientX - box.left) * element.width / box.width - view.x) / view.scale,
    y: ((event.clientY - box.top) * element.height / box.height - view.y) / view.scale,
  }
}

function moveCrop(x, y) {
  const { source, crop } = preview.value
  const next = clampCrop(source, { x, y }, props.cropSize)
  if (crop.x === next.x && crop.y === next.y) return
  preview.value.crop = next
  emit('change')
  paint()
}

function startDrag(event) {
  if (props.disabled || !preview.value || !view || drag) return
  const { x, y } = imagePoint(event)
  const { crop } = preview.value
  const size = props.cropSize
  if (x < crop.x || x > crop.x + size || y < crop.y || y > crop.y + size) {
    moveCrop(x - size / 2, y - size / 2)
  }
  const origin = preview.value.crop
  drag = { id: event.pointerId, x: x - origin.x, y: y - origin.y }
  canvas.value.setPointerCapture(event.pointerId)
}

function updateDrag(event) {
  if (!drag || event.pointerId !== drag.id || props.disabled || !view) return
  const { x, y } = imagePoint(event)
  moveCrop(x - drag.x, y - drag.y)
}

function endDrag(event) {
  if (event.pointerId === drag?.id) drag = null
}

watch(() => props.disabled, () => { drag = null })
watch([() => props.file, () => props.cropSize], async ([file, size], _, onCleanup) => {
  preview.value = null
  drag = null
  paint()
  emit('ready', false)
  if (!file || !size) return
  let current = true
  const url = URL.createObjectURL(file)
  onCleanup(() => {
    current = false
    URL.revokeObjectURL(url)
  })

  const image = new Image()
  image.src = url
  try {
    await image.decode()
  } catch {
    if (current) emit('error', 'The browser could not decode this image.')
    return
  } finally {
    URL.revokeObjectURL(url)
  }
  if (!current) return

  try {
    const source = await readPngLabels(file)
    if (!current) return
    if (source.width < size || source.height < size) {
      emit('error', `Image must be at least ${size} × ${size}.`)
      return
    }
    preview.value = {
      source, image,
      crop: { x: Math.floor((source.width - size) / 2), y: Math.floor((source.height - size) / 2) },
    }
    paint()
    emit('ready', true)
  } catch (reason) {
    if (current) emit('error', reason instanceof Error ? reason.message : 'Could not read PNG labels.')
  }
})

onMounted(() => {
  observer = new ResizeObserver(paint)
  observer.observe(canvas.value)
})
onBeforeUnmount(() => observer?.disconnect())
defineExpose({
  getAnchorImage() {
    const section = preview.value
    return section && cropAnchor(section.source, section.crop, props.cropSize)
  },
})
</script>

<template>
  <section class="panel">
    <header class="panel-header"><h2>Image</h2></header>
    <div class="canvas-wrap" :class="{ 'crop-canvas--disabled': disabled }">
      <canvas ref="canvas" @pointerdown="startDrag" @pointermove="updateDrag"
        @pointerup="endDrag" @pointercancel="endDrag" @lostpointercapture="endDrag" />
      <div v-if="!preview" class="empty-state"><strong>No section selected</strong></div>
    </div>
  </section>
</template>
