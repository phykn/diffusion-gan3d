<script setup>
import { computed, ref } from 'vue'
import { MIN_BLOCKS, anchorNote, normalizeBlock, normalizeSeed } from '../input-state.js'

const props = defineProps({
  file: File,
  health: Object,
  healthError: String,
  busy: Boolean,
  ready: Boolean,
  error: String,
  porosity: Number,
  tortuosity: Number,
})
const seed = defineModel('seed', { type: Number })
const blocks = defineModel('blocks', { type: Array })
const axis = defineModel('axis', { type: Number })
const heightOrigin = defineModel('heightOrigin', { type: Number })
const heightExtent = defineModel('heightExtent', { type: Number })
const measureTortuosity = defineModel('measureTortuosity', { type: Boolean })
const emit = defineEmits(['select-file', 'generate'])
const dragging = ref(false)
const blockLimit = computed(() => props.health?.max_axis_blocks ?? MIN_BLOCKS)
const blockNote = computed(() => anchorNote(blocks.value, props.health?.crop_size))
const metrics = computed(() => [
  ['Porosity', Number.isFinite(props.porosity) ? `${(props.porosity * 100).toFixed(2)}%` : '—'],
  ['Tortuosity', props.tortuosity === Infinity ? '∞' : Number.isFinite(props.tortuosity) ? props.tortuosity.toFixed(3) : '—'],
])

function select(files) {
  dragging.value = false
  if (!props.busy && files?.length) emit('select-file', files[0])
}

function upload(event) {
  select(event.target.files)
  event.target.value = ''
}

function updateSeed(input) {
  seed.value = normalizeSeed(input.value)
  input.value = String(seed.value)
}

function updateBlock(axis, input) {
  const value = normalizeBlock(input.value, blockLimit.value)
  blocks.value = blocks.value.map((previous, index) => index === axis ? value : previous)
  input.value = String(value)
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-section">
      <label
        class="upload-button"
        :class="{ 'upload-button--dragging': dragging, 'upload-button--disabled': busy }"
        @dragenter.prevent="!busy && (dragging = true)"
        @dragover.prevent
        @dragleave.prevent="dragging = false"
        @drop.prevent="select($event.dataTransfer.files)"
      >
        <input
          class="visually-hidden"
          type="file"
          accept="image/png"
          :disabled="busy"
          @change="upload"
        >
        <strong>{{ file?.name || 'Upload image' }}</strong>
      </label>
    </div>
    <label class="number-control">
      <span>Section plane</span>
      <select v-model.number="axis" :disabled="busy">
        <option :value="0">XY · rows Y</option>
        <option :value="1">XZ · rows Z</option>
        <option :value="2">YZ · rows Z</option>
      </select>
    </label>
    <template v-if="health?.height_enabled && axis === 0">
      <label class="number-control">
        <span>Z origin (px)</span>
        <input v-model.number="heightOrigin" type="number" min="0" :disabled="busy">
      </label>
      <label class="number-control">
        <span>Full Z extent (px)</span>
        <input v-model.number="heightExtent" type="number" min="1" :disabled="busy">
      </label>
    </template>
    <p v-else-if="health?.height_enabled" class="status">Z origin follows the crop rows. Full Z extent is the uploaded image height.</p>
    <div class="blocks-control">
      <span>Blocks</span>
      <label v-for="(name, index) in ['Z', 'Y', 'X']" :key="name">
        <small>{{ name }}</small>
        <input
          :value="blocks[index]"
          type="number"
          :min="MIN_BLOCKS"
          :max="blockLimit"
          step="1"
          :disabled="busy"
          @change="updateBlock(index, $event.target)"
        >
      </label>
    </div>
    <p v-if="blockNote" class="status">{{ blockNote }}</p>
    <label class="number-control">
      <span>Seed</span>
      <input :value="seed" type="number" min="0" step="1" :disabled="busy" @change="updateSeed($event.target)">
    </label>
    <label class="number-control" title="Solved on the server before the volume is sent; slow for large volumes.">
      <span>Tortuosity</span>
      <input v-model="measureTortuosity" class="toggle" type="checkbox" :disabled="busy">
    </label>
    <button class="primary-button" type="button" :disabled="!ready" @click="emit('generate')">
      {{ busy ? 'RUNNING…' : 'RUN' }}
    </button>
    <p v-if="error" class="status status--error">{{ error }}</p>
    <dl class="metrics">
      <div v-for="[label, value] in metrics" :key="label">
        <dt>{{ label }}</dt>
        <dd>{{ value }}</dd>
      </div>
    </dl>
    <div class="server-state" :class="{ ready: health }">
      <span></span>{{ health ? `Inference ready · ${health.device}` : healthError || 'Connecting to inference…' }}
    </div>
  </aside>
</template>
