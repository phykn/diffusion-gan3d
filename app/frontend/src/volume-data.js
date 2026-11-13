import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray.js'
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData.js'

export function replaceVolumeData(mapper, previous, labels, shape) {
  if (previous) {
    mapper?.setInputData(null)
    previous.getPointData().getScalars()?.delete()
    previous.delete()
  }
  if (!labels || !shape) return null

  const [z, y, x] = shape
  const data = vtkImageData.newInstance()
  data.setDimensions(x, y, z)
  data.getPointData().setScalars(vtkDataArray.newInstance({ name: 'phase', values: labels }))
  mapper.setInputData(data)
  return data
}
