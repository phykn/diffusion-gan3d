<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

import '@kitware/vtk.js/Rendering/Profiles/Volume'
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray'
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction'
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData'
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane'
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction'
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume'
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper'
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow'

const props = defineProps({ values: Uint8Array, shape: Array, busy: Boolean })
const host = ref(null)
const phase0Visible = ref(true)
const phase1Visible = ref(true)
const phase0Color = ref('#25282a')
const phase1Color = ref('#d98266')
const clipDepth = ref(0)
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
  const phase0 = hexToRgb(phase0Color.value)
  const phase1 = hexToRgb(phase1Color.value)
  const transitionStart = .35
  const transitionEnd = .65
  colorTransfer ??= vtkColorTransferFunction.newInstance()
  colorTransfer.removeAllPoints()
  colorTransfer.addRGBPoint(0, ...phase0)
  colorTransfer.addRGBPoint(transitionStart, ...phase0)
  colorTransfer.addRGBPoint(transitionEnd, ...phase1)
  colorTransfer.addRGBPoint(1, ...phase1)

  opacityTransfer ??= vtkPiecewiseFunction.newInstance()
  opacityTransfer.removeAllPoints()
  const phase0Opacity = phase0Visible.value ? 1 : 0
  const phase1Opacity = phase1Visible.value ? 1 : 0
  opacityTransfer.addPoint(0, phase0Opacity)
  opacityTransfer.addPoint(transitionStart, phase0Opacity)
  opacityTransfer.addPoint(transitionEnd, phase1Opacity)
  opacityTransfer.addPoint(1, phase1Opacity)

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
  clipDepth.value = Math.max(0, depth - 1)
  updateClipDepth()
  setDefaultView()
}

function updateClipDepth() {
  clippingPlane?.setOrigin(0, 0, Number(clipDepth.value))
  mapper?.modified()
  renderWindow?.render()
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
function reset() { setCamera(true) }
function togglePhase(phase) {
  if (phase === 0) phase0Visible.value = !phase0Visible.value
  else phase1Visible.value = !phase1Visible.value
  updateAppearance()
}

watch(() => props.values, values => setVolume(values, props.shape))
onMounted(() => {
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
        <div class="phase-control" :class="{ active: phase0Visible }">
          <button type="button" :aria-pressed="phase0Visible" @click="togglePhase(0)">Phase 0</button>
          <label title="Phase 0 color"><input v-model="phase0Color" type="color" aria-label="Phase 0 color" @input="updateAppearance"><i :style="{ background: phase0Color }"></i></label>
        </div>
        <div class="phase-control" :class="{ active: phase1Visible }">
          <button type="button" :aria-pressed="phase1Visible" @click="togglePhase(1)">Phase 1</button>
          <label title="Phase 1 color"><input v-model="phase1Color" type="color" aria-label="Phase 1 color" @input="updateAppearance"><i :style="{ background: phase1Color }"></i></label>
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
        <span>Axis 0</span>
        <input v-model.number="clipDepth" type="range" min="0" :max="Math.max(0, (shape?.[0] ?? 1) - 1)" step="1" :disabled="!values" @input="updateClipDepth">
        <output>{{ clipDepth }}</output>
      </label>
      <button type="button" :disabled="!values" @click="reset">Reset view</button>
    </footer>
  </section>
</template>
