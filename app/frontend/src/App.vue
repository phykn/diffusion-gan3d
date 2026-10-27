<script setup>
import { defineAsyncComponent, ref } from 'vue'
import CropEditor from './components/CropEditor.vue'
import Sidebar from './components/Sidebar.vue'
import { useGeneration } from './use-generation.js'

const VolumeViewer = defineAsyncComponent(() => import('./components/VolumeViewer.vue'))
const cropEditor = ref(null)
const {
  cropReady,
  file,
  seed,
  blocks,
  axis,
  domain,
  heightOrigin,
  heightExtent,
  busy,
  error,
  health,
  healthError,
  measureTortuosity,
  result,
  ready,
  invalidateResult,
  selectFile,
  generate,
} = useGeneration(cropEditor)
</script>

<template>
  <main class="app">
    <Sidebar
      v-model:seed="seed"
      v-model:blocks="blocks"
      v-model:axis="axis"
      v-model:domain="domain"
      v-model:height-origin="heightOrigin"
      v-model:height-extent="heightExtent"
      v-model:measure-tortuosity="measureTortuosity"
      :file
      :health
      :health-error="healthError"
      :busy
      :ready
      :error
      :porosity="result?.porosity"
      :tortuosity="result?.tortuosity"
      @select-file="selectFile"
      @generate="generate"
    />
    <section class="editor-main">
      <div class="workspace">
        <CropEditor
          ref="cropEditor"
          :file
          :crop-size="health?.crop_size"
          :disabled="busy"
          @change="invalidateResult"
          @error="error = $event"
          @ready="cropReady = $event"
        />
        <VolumeViewer
          :values="result?.values"
          :shape="result?.shape"
          :busy
          :num-phases="health?.num_phases"
        />
      </div>
    </section>
  </main>
</template>
