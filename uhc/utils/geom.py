from vtk import (
    vtkQuadricDecimation,
    vtkPolyData,
    vtkSTLReader,
    vtkSTLWriter,
    vtkTransform,
    vtkCenterOfMass,
    vtkTransformPolyDataFilter,
)


def quadric_mesh_decimation(fname, reduction_rate, verbose=False):
    reader = vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    inputPoly = reader.GetOutput()

    decimate = vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(reduction_rate)
    decimate.Update()
    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print(
            f"Mesh Decimation: (points, faces) goes from ({inputPoly.GetNumberOfPoints(), inputPoly.GetNumberOfPolys()}) "
            f"to ({decimatedPoly.GetNumberOfPoints(), decimatedPoly.GetNumberOfPolys()})"
        )

    stlWriter = vtkSTLWriter()
    stlWriter.SetFileName(fname)
    stlWriter.SetFileTypeToBinary()
    stlWriter.SetInputData(decimatedPoly)
    stlWriter.Write()


def center_scale_mesh(fname, scale):
    reader = vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    inputPoly = reader.GetOutputPort()

    centerOfMassFilter = vtkCenterOfMass()
    centerOfMassFilter.SetInputConnection(inputPoly)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = centerOfMassFilter.GetCenter()

    transform = vtkTransform()
    transform.PostMultiply()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(scale, scale, scale)
    transform.Translate(center[0], center[1], center[2])
    transform.Update()

    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(inputPoly)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    stlWriter = vtkSTLWriter()
    stlWriter.SetFileName(fname)
    stlWriter.SetFileTypeToBinary()
    stlWriter.SetInputConnection(transformFilter.GetOutputPort())
    stlWriter.Write()