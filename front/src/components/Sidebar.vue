<script setup>
import { ref } from 'vue'
import { MAX_BLOCKS, MIN_BLOCKS, normalizeBlock, normalizeSeed } from '../input-state.js'

const props = defineProps({
  file: File,
  health: Object,
  busy: Boolean,
  ready: Boolean,
  status: String,
  error: Boolean,
  porosity: Number,
  tortuosity: Number,
})
const seed = defineModel('seed', { type: Number, default: 0 })
const blocks = defineModel('blocks', { type: Array, default: () => [1, 1, 1] })
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
  const value = normalizeBlock(event.target.value)
  const next = [...blocks.value]
  next[index] = value
  blocks.value = next
  event.target.value = String(value)
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-section">
      <label class="upload-button" :class="{ 'upload-button--dragging': dragging, 'upload-button--disabled': busy }" @dragenter.prevent="!busy && (dragging = true)" @dragover.prevent @dragleave.prevent="dragging = false" @drop.prevent="accept($event.dataTransfer.files)">
        <input class="visually-hidden" type="file" accept="image/png,image/jpeg" :disabled="busy" @change="selectInput">
        <strong>{{ file?.name || 'Upload image' }}</strong>
      </label>
    </div>
    <label class="number-control"><span>Seed</span><input :value="seed" type="number" min="0" step="1" :disabled="busy" @input="setSeed"></label>
    <div class="blocks-control"><span>Blocks</span><label v-for="(axis, index) in ['X', 'Y', 'Z']" :key="axis"><small>{{ axis }}</small><input :value="blocks[index]" type="number" :min="MIN_BLOCKS" :max="MAX_BLOCKS" step="1" :disabled="busy" @input="setBlock(index, $event)"></label></div>
    <button class="primary-button" type="button" :disabled="!ready" @click="emit('generate')">{{ busy ? 'RUNNING…' : 'RUN' }}</button>
    <p v-if="error" class="status status--error">{{ status }}</p>
    <dl class="metrics">
      <div><dt>Porosity</dt><dd>{{ Number.isFinite(porosity) ? `${(porosity * 100).toFixed(2)}%` : '—' }}</dd></div>
      <div><dt>Tortuosity</dt><dd>{{ Number.isFinite(tortuosity) ? tortuosity.toFixed(3) : '—' }}</dd></div>
    </dl>
    <div class="server-state" :class="{ ready: health }"><span></span>{{ health ? `Inference ready · ${health.device}` : 'Connecting to inference…' }}</div>
  </aside>
</template>
