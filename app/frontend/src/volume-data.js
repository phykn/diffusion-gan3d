import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray.js'
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData.js'

export function replaceVolumeData(mapper, previous, values, shape) {
  mapper?.setInputData(null)
  previous?.getPointData().getScalars()?.delete()
  previous?.delete()
  if (!values || !shape) return null

  const [depth, height, width] = shape
  const image = vtkImageData.newInstance()
  image.setDimensions(width, height, depth)
  image.setSpacing(1, 1, 1)
  image.getPointData().setScalars(vtkDataArray.newInstance({
    name: 'phase',
    numberOfComponents: 1,
    values,
  }))
  mapper.setInputData(image)
  return image
}
