import assert from 'node:assert/strict'
import test from 'node:test'

import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper.js'

import { replaceVolumeData } from '../src/volume-data.js'

test('clearing a volume detaches the mapper and releases the label buffer', () => {
  const mapper = vtkVolumeMapper.newInstance()
  const values = new Uint8Array(24)
  let image = replaceVolumeData(mapper, null, values, [2, 3, 4])
  const scalars = image.getPointData().getScalars()
  const previous = image
  assert.equal(mapper.getInputData(), image)
  assert.deepEqual(image.getDimensions(), [4, 3, 2])
  assert.equal(scalars.getData(), values)

  image = replaceVolumeData(mapper, image, null, null)
  assert.equal(image, null)
  assert.equal(mapper.getInputData(), null)
  assert.equal(previous.isDeleted(), true)
  assert.equal(scalars.isDeleted(), true)
  assert.equal(replaceVolumeData(mapper, image, null, null), null)
  mapper.delete()
})

test('replacing a volume releases its predecessor and installs the new data', () => {
  const mapper = vtkVolumeMapper.newInstance()
  const first = replaceVolumeData(mapper, null, new Uint8Array(8), [2, 2, 2])
  const firstScalars = first.getPointData().getScalars()
  const values = new Uint8Array([1, 2, 3])
  const second = replaceVolumeData(mapper, first, values, [1, 1, 3])
  assert.equal(first.isDeleted(), true)
  assert.equal(firstScalars.isDeleted(), true)
  assert.equal(mapper.getInputData(), second)
  assert.deepEqual(second.getDimensions(), [3, 1, 1])
  assert.equal(second.getPointData().getScalars().getData(), values)
  replaceVolumeData(mapper, second, null, null)
  mapper.delete()
})
