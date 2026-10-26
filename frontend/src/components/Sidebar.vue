<script setup>
import { ref } from 'vue'
import { MIN_BLOCKS, normalizeBlock, normalizeSeed } from '../input-state.js'

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
const domain = defineModel('domain', { type: Number })
const heightOrigin = defineModel('heightOrigin', { type: Number })
const heightExtent = defineModel('heightExtent', { type: Number })
const measureTortuosity = defineModel('measureTortuosity', { type: Boolean })
const emit = defineEmits(['select-file', 'generate'])
const dragging = ref(false)

function accept(files) {
  if (props.busy) return
  const file = files?.[0]
  if (file) emit('select-file', file)
  dragging.value = false
}

function selectInput(event) {
  if (props.busy) return
  accept(event.target.files)
  event.target.value = ''
}

function setSeed(event) {
  const value = normalizeSeed(event.target.value)
  seed.value = value
  event.target.value = String(value)
}

function setBlock(index, event) {
  const value = normalizeBlock(event.target.value, props.health?.max_axis_blocks ?? MIN_BLOCKS)
  const next = [...blocks.value]
  next[index] = value
  blocks.value = next
  event.target.value = String(value)
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
        @drop.prevent="accept($event.dataTransfer.files)"
      >
        <input
          class="visually-hidden"
          type="file"
          accept="image/png"
          :disabled="busy"
          @change="selectInput"
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
    <label v-if="health?.num_domains > 1" class="number-control">
      <span>Domain</span>
      <select v-model.number="domain" :disabled="busy">
        <option v-for="n in health.num_domains" :key="n" :value="n - 1">{{ n - 1 }}</option>
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
          :max="health?.max_axis_blocks ?? MIN_BLOCKS"
          step="1"
          :disabled="busy"
          @change="setBlock(index, $event)"
        >
      </label>
    </div>
    <label class="number-control">
      <span>Seed</span>
      <input :value="seed" type="number" min="0" step="1" :disabled="busy" @change="setSeed">
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
      <div>
        <dt>Porosity</dt>
        <dd>{{ Number.isFinite(porosity) ? `${(porosity * 100).toFixed(2)}%` : '—' }}</dd>
      </div>
      <div>
        <dt>Tortuosity</dt>
        <dd>{{ tortuosity === Infinity ? '∞' : Number.isFinite(tortuosity) ? tortuosity.toFixed(3) : '—' }}</dd>
      </div>
    </dl>
    <div class="server-state" :class="{ ready: health }">
      <span></span>{{ health ? `Inference ready · ${health.device}` : healthError || 'Connecting to inference…' }}
    </div>
  </aside>
</template>
