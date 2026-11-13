<script setup>
import { defineAsyncComponent, proxyRefs, ref } from 'vue'
import CropEditor from './components/CropEditor.vue'
import Sidebar from './components/Sidebar.vue'
import { useGeneration } from './use-generation.js'

const VolumeViewer = defineAsyncComponent(() => import('./components/VolumeViewer.vue'))
const editor = ref(null)
const form = proxyRefs(useGeneration(editor))
</script>

<template>
  <main class="app">
    <Sidebar
      v-model:seed="form.seed"
      v-model:blocks="form.blocks"
      v-model:axis="form.axis"
      v-model:height-origin="form.heightOrigin"
      v-model:height-extent="form.heightExtent"
      v-model:measure-tortuosity="form.measureTortuosity"
      :file="form.file" :health="form.health" :health-error="form.healthError"
      :busy="form.busy" :ready="form.ready" :error="form.error"
      :porosity="form.result?.porosity" :tortuosity="form.result?.tortuosity"
      @select-file="form.selectFile" @generate="form.generate"
    />
    <section class="editor-main">
      <div class="workspace">
        <CropEditor
          ref="editor" :file="form.file" :crop-size="form.health?.crop_size" :disabled="form.busy"
          @change="form.invalidateResult" @error="form.error = $event" @ready="form.cropReady = $event"
        />
        <VolumeViewer
          :values="form.result?.values" :shape="form.result?.shape"
          :busy="form.busy" :num-phases="form.health?.num_phases"
        />
      </div>
    </section>
  </main>
</template>
