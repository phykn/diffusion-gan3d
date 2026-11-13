<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import '@kitware/vtk.js/Rendering/Profiles/Volume'
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction'
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane'
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction'
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume'
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper'
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow'

import { replaceVolumeData } from '../volume-data.js'

const props = defineProps({
  values: Uint8Array,
  shape: Array,
  busy: Boolean,
  numPhases: { type: Number, default: 2 },
})
const host = ref(null)
const palette = ['#25282a', '#d98266', '#5f8d8a', '#d6a84f', '#8c6bb1', '#77945a']
const phases = ref([])
const clipDepth = ref(0)
const clipDepthMax = computed(() => Math.max(0, (props.shape?.[0] ?? 1) - 1))
let scene

function createScene(container) {
  const view = vtkGenericRenderWindow.newInstance({
    background: [.973, .965, .941],
    listenWindowResize: false,
  })
  const mapper = vtkVolumeMapper.newInstance({
    sampleDistance: .45,
    imageSampleDistance: .7,
    autoAdjustSampleDistances: false,
    maximumSamplesPerRay: 2000,
    initialInteractionScale: 1,
    interactionSampleDistanceFactor: 1,
  })
  // Keep the exposed cut face toward both camera positions on the negative z side.
  const clip = vtkPlane.newInstance({ normal: [0, 0, 1], origin: [0, 0, 0] })
  const color = vtkColorTransferFunction.newInstance()
  const opacity = vtkPiecewiseFunction.newInstance()
  const actor = vtkVolume.newInstance()
  mapper.addClippingPlane(clip)
  actor.setMapper(mapper)
  actor.setVisibility(false)
  const property = actor.getProperty()
  property.setRGBTransferFunction(0, color)
  property.setScalarOpacity(0, opacity)
  property.setScalarOpacityUnitDistance(0, .85)
  property.setInterpolationTypeToNearest()
  property.setShade(true)
  property.setAmbient(.72)
  property.setDiffuse(.36)
  property.setSpecular(0)

  view.setContainer(container)
  const renderer = view.getRenderer()
  renderer.addVolume(actor)
  const resize = new ResizeObserver(() => view.resize())
  resize.observe(container)
  view.resize()
  return { view, mapper, clip, color, opacity, actor, renderer, resize, data: null }
}

function render() {
  scene.view.getRenderWindow().render()
}

function syncPhases() {
  const count = Math.max(1, Math.floor(Number(props.numPhases) || 2))
  phases.value = Array.from({ length: count }, (_, i) => phases.value[i] ?? {
    color: palette[i % palette.length], visible: true,
  })
  if (scene) paintPhases()
}

function paintPhases() {
  const { color, opacity } = scene
  color.removeAllPoints()
  opacity.removeAllPoints()
  const last = phases.value.length - 1
  phases.value.forEach((phase, i) => {
    const rgb = [1, 3, 5].map(offset => parseInt(phase.color.slice(offset, offset + 2), 16) / 255)
    for (const label of [i === 0 ? 0 : i - .35, i === last ? i : i + .35]) {
      color.addRGBPoint(label, ...rgb)
      opacity.addPoint(label, Number(phase.visible))
    }
  })
  render()
}

function setVolume(values, shape) {
  if (!scene) return
  scene.data = replaceVolumeData(scene.mapper, scene.data, values, shape)
  scene.actor.setVisibility(Boolean(scene.data))
  if (scene.data) positionCamera(shape, false)
  else render()
}

function positionCamera(shape, front) {
  const [depth, height, width] = shape
  const center = [width, height, depth].map(size => (size - 1) / 2)
  const extent = Math.max(...shape)
  const offsets = front ? [0, 0, -2.5] : [1.6, -1.05, -1.8]
  const camera = scene.renderer.getActiveCamera()
  camera.setParallelProjection(true)
  camera.setFocalPoint(...center)
  camera.setPosition(...center.map((value, axis) => value + offsets[axis] * extent))
  camera.setViewUp(0, -1, 0)
  camera.setParallelScale(extent * (front ? .56 : .72))
  scene.renderer.resetCameraClippingRange()
  setClipDepth(0)
}

function setClipDepth(depth) {
  clipDepth.value = depth
  scene.clip.setOrigin(0, 0, depth)
  scene.mapper.modified()
  render()
}

watch(() => props.numPhases, syncPhases, { immediate: true })
watch([() => props.values, () => props.shape], ([values, shape]) => setVolume(values, shape))
onMounted(() => {
  scene = createScene(host.value)
  paintPhases()
  setVolume(props.values, props.shape)
})
onBeforeUnmount(() => {
  if (!scene) return
  const { resize, renderer, actor, mapper, data, clip, color, opacity, view } = scene
  resize.disconnect()
  renderer.removeVolume(actor)
  replaceVolumeData(mapper, data, null, null)
  for (const object of [actor, mapper, clip, color, opacity, view]) object.delete()
  scene = null
})
</script>

<template>
  <section class="panel">
    <header class="panel-header">
      <div><h2>3D Generated</h2></div>
      <div class="phase-toggles">
        <div
          v-for="(entry, phase) in phases"
          :key="phase"
          class="phase-control"
          :class="{ active: entry.visible }"
        >
          <button
            type="button"
            :aria-pressed="entry.visible"
            @click="entry.visible = !entry.visible; paintPhases()"
          >Phase {{ phase }}</button>
          <label :title="`Phase ${phase} color`">
            <input
              :value="entry.color"
              type="color"
              :aria-label="`Phase ${phase} color`"
              @input="entry.color = $event.target.value; paintPhases()"
            >
            <i :style="{ background: entry.color }"></i>
          </label>
        </div>
      </div>
    </header>
    <div ref="host" class="canvas-wrap">
      <div v-if="!values && !busy" class="empty-state">
        <strong>No generated volume</strong>
      </div>
      <div v-if="busy" class="loading">
        <span></span><strong>Generating…</strong>
      </div>
    </div>
    <footer>
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
          @input="setClipDepth(Number($event.target.value))"
        >
        <output>{{ clipDepth }}</output>
      </label>
      <button type="button" :disabled="!values" @click="positionCamera(shape, true)">Reset</button>
    </footer>
  </section>
</template>
