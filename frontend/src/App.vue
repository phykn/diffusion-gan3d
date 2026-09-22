<script setup>
import { defineAsyncComponent, ref } from 'vue'
import CropEditor from './components/CropEditor.vue'
import Sidebar from './components/Sidebar.vue'
import { useGeneration } from './use-generation.js'

const VolumeViewer = defineAsyncComponent(() => import('./components/VolumeViewer.vue'))
const cropEditor = ref(null)
const {
  cropReady, file, seed, blocks, plane, domain, heightOrigin, heightExtent, busy, status, error, health, result, ready,
  invalidateResult, selectFile, generate,
} = useGeneration(cropEditor)
</script>

<template>
  <main class="app">
    <Sidebar v-model:seed="seed" v-model:blocks="blocks" v-model:plane="plane" v-model:domain="domain" v-model:height-origin="heightOrigin" v-model:height-extent="heightExtent" :file :health :busy :ready :status :error :porosity="result?.porosity" :tortuosity="result?.tortuosity" @select-file="selectFile" @generate="generate" />
    <section class="editor-main">
      <div class="workspace">
        <CropEditor
          ref="cropEditor"
          :file
          :crop-size="health?.crop_size ?? 128"
          :disabled="busy"
          @change="invalidateResult"
          @error="status = $event; error = true"
          @ready="cropReady = $event"
        />
        <VolumeViewer :values="result?.values" :shape="result?.shape" :busy :num-phases="health?.num_phases ?? 2" />
      </div>
    </section>
  </main>
</template>
