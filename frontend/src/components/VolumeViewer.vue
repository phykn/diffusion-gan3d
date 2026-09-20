<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import '@kitware/vtk.js/Rendering/Profiles/Volume'
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray'
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction'
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData'
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane'
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction'
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume'
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper'
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow'

const props = defineProps({ values: Uint8Array, shape: Array, busy: Boolean, numPhases: { type: Number, default: 2 } })
const host = ref(null)
const defaultPhaseColors = ['#25282a', '#d98266', '#5f8d8a', '#d6a84f', '#8c6bb1', '#77945a']
const phaseVisibility = ref([])
const phaseColors = ref([])
const clipDepth = ref(0)
const clipDepthMax = computed(() => Math.max(0, (props.shape?.[0] ?? 1) - 1))
let view
let renderer
let renderWindow
let actor
let mapper
let imageData
let clippingPlane
let colorTransfer
let opacityTransfer
let resizeObserver

function ensurePhaseState() {
  const count = Math.max(1, Math.floor(Number(props.numPhases) || 2))
  phaseVisibility.value = Array.from({ length: count }, (_, index) => phaseVisibility.value[index] ?? true)
  phaseColors.value = Array.from({ length: count }, (_, index) => phaseColors.value[index] ?? defaultPhaseColors[index % defaultPhaseColors.length])
}

function initialize() {
  view = vtkGenericRenderWindow.newInstance({ background: [.973, .965, .941], listenWindowResize: false })
  view.setContainer(host.value)
  renderer = view.getRenderer()
  renderWindow = view.getRenderWindow()
  mapper = vtkVolumeMapper.newInstance({
    sampleDistance: .45,
    imageSampleDistance: .7,
    autoAdjustSampleDistances: false,
    maximumSamplesPerRay: 2000,
    initialInteractionScale: 1,
    interactionSampleDistanceFactor: 1,
  })
  clippingPlane = vtkPlane.newInstance({ normal: [0, 0, -1], origin: [0, 0, 0] })
  mapper.addClippingPlane(clippingPlane)
  actor = vtkVolume.newInstance()
  actor.setMapper(mapper)
  configureAppearance()
  renderer.addVolume(actor)
  resizeObserver = new ResizeObserver(() => view.resize())
  resizeObserver.observe(host.value)
  view.resize()
}

function configureAppearance() {
  ensurePhaseState()
  colorTransfer ??= vtkColorTransferFunction.newInstance()
  colorTransfer.removeAllPoints()
  opacityTransfer ??= vtkPiecewiseFunction.newInstance()
  opacityTransfer.removeAllPoints()
  const lastPhase = phaseColors.value.length - 1
  for (let phase = 0; phase <= lastPhase; phase += 1) {
    const start = phase === 0 ? 0 : phase - .35
    const end = phase === lastPhase ? lastPhase : phase + .35
    const color = hexToRgb(phaseColors.value[phase])
    const opacity = phaseVisibility.value[phase] ? 1 : 0
    colorTransfer.addRGBPoint(start, ...color)
    colorTransfer.addRGBPoint(end, ...color)
    opacityTransfer.addPoint(start, opacity)
    opacityTransfer.addPoint(end, opacity)
  }

  const property = actor.getProperty()
  property.setRGBTransferFunction(0, colorTransfer)
  property.setScalarOpacity(0, opacityTransfer)
  property.setScalarOpacityUnitDistance(0, .85)
  property.setInterpolationTypeToLinear()
  property.setShade(true)
  property.setAmbient(.72)
  property.setDiffuse(.36)
  property.setSpecular(0)
}

function hexToRgb(hex) {
  const value = Number.parseInt(hex.slice(1), 16)
  return [(value >> 16) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255]
}

function updateAppearance() {
  configureAppearance()
  renderWindow?.render()
}

function setVolume(values, shape) {
  if (!values || !shape) {
    actor?.setVisibility(false)
    renderWindow?.render()
    return
  }
  const [depth, height, width] = shape
  imageData?.delete()
  imageData = vtkImageData.newInstance()
  imageData.setDimensions(width, height, depth)
  imageData.setSpacing(1, 1, 1)
  imageData.getPointData().setScalars(vtkDataArray.newInstance({
    name: 'phase',
    numberOfComponents: 1,
    values,
  }))
  mapper.setInputData(imageData)
  actor.setVisibility(true)
  resetClipDepth(shape)
  setDefaultView()
}

function updateClipDepth() {
  clippingPlane?.setOrigin(0, 0, Number(clipDepth.value))
  mapper?.modified()
  renderWindow?.render()
}

function resetClipDepth(shape) {
  clipDepth.value = Math.max(0, (shape?.[0] ?? 1) - 1)
  updateClipDepth()
}

function setClipDepth(event) {
  clipDepth.value = Number(event.target.value)
  updateClipDepth()
}

function setCamera(front) {
  if (!renderer || !props.shape) return
  const [depth, height, width] = props.shape
  const camera = renderer.getActiveCamera()
  const center = [(width - 1) / 2, (height - 1) / 2, (depth - 1) / 2]
  const extent = Math.max(width, height, depth)
  camera.setParallelProjection(true)
  camera.setFocalPoint(...center)
  camera.setPosition(...(front
    ? [center[0], center[1], -extent * 2.5]
    : [center[0] + extent * 1.6, center[1] - extent * 1.05, center[2] - extent * 1.8]))
  camera.setViewUp(0, -1, 0)
  camera.setParallelScale(extent * (front ? .56 : .72))
  renderer.resetCameraClippingRange()
  renderWindow.render()
}

function setDefaultView() { setCamera(false) }
function reset() {
  resetClipDepth(props.shape)
  setCamera(true)
}
function togglePhase(phase) {
  phaseVisibility.value = phaseVisibility.value.map((visible, index) => index === phase ? !visible : visible)
  updateAppearance()
}
function setPhaseColor(phase, event) {
  phaseColors.value = phaseColors.value.map((color, index) => index === phase ? event.target.value : color)
  updateAppearance()
}

watch(() => props.values, values => setVolume(values, props.shape))
watch(() => props.shape?.[0], () => resetClipDepth(props.shape), { immediate: true, flush: 'sync' })
watch(() => props.numPhases, () => {
  ensurePhaseState()
  updateAppearance()
})
onMounted(() => {
  ensurePhaseState()
  initialize()
  if (props.values) setVolume(props.values, props.shape)
})
onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  imageData?.delete()
  mapper?.delete()
  clippingPlane?.delete()
  actor?.delete()
  colorTransfer?.delete()
  opacityTransfer?.delete()
  view?.delete()
})
</script>

<template>
  <section class="panel">
    <header class="panel-header">
      <div><h2>3D Generated</h2></div>
      <div class="phase-toggles">
        <div v-for="(color, phase) in phaseColors" :key="phase" class="phase-control" :class="{ active: phaseVisibility[phase] }">
          <button type="button" :aria-pressed="phaseVisibility[phase]" @click="togglePhase(phase)">Phase {{ phase }}</button>
          <label :title="`Phase ${phase} color`"><input :value="color" type="color" :aria-label="`Phase ${phase} color`" @input="setPhaseColor(phase, $event)"><i :style="{ background: color }"></i></label>
        </div>
      </div>
    </header>
    <div ref="host" class="canvas-wrap volume-canvas">
      <div v-if="!values && !busy" class="empty-state"><strong>No generated volume</strong></div>
      <div v-if="busy" class="loading"><span></span><strong>Generating…</strong></div>
    </div>
    <footer class="volume-footer">
      <span>{{ shape ? shape.join(' × ') : '—' }}</span>
      <label class="depth-control">
        <input
          :key="`clip-depth-${shape?.[0] ?? 0}`"
          :value="clipDepth"
          type="range"
          min="0"
          :max="clipDepthMax"
          step="1"
          :disabled="!values"
          @input="setClipDepth"
        >
        <output>{{ clipDepth }}</output>
      </label>
      <button type="button" :disabled="!values" @click="reset">Reset</button>
    </footer>
  </section>
</template>
