# trace generated using paraview version 5.13.3
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
# numpy
import numpy as np
# os
import os

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# i?��e?? i�׀i??
folder_path = "C:/Users/geono/Documents/shear/250630_practice/pulsatile/vel"

# e����e?? xdmf file
xdmf_files = [f for f in os.listdir(folder_path) if f.endswith(".xdmf")]

# e��?e����
for file_name in xdmf_files:
    full_path = os.path.join(folder_path, file_name).replace("\\", "/")
    base_name = os.path.splitext(file_name)[0]
    reader = Xdmf3ReaderS(registrationName=file_name, FileName=[full_path])

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    readerDisplay = Show(reader, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    readerDisplay.Representation = 'Surface'

    # reset view to fit data
    renderView1.ResetCamera(False, 0.9)

    # changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.0, 1.25, 167.5]
    renderView1.CameraFocalPoint = [0.0, 1.25, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    readerDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(readerDisplay, ('POINTS', 'u', 'Magnitude'))

	# rescale color and/or opacity maps used to include current data range
    readerDisplay.RescaleTransferFunctionToDataRange(True, False)

	# show color bar/color legend
    readerDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'u'
    lLUT = GetColorTransferFunction('u')

    # get opacity transfer function/opacity map for 'u'
    lPWF = GetOpacityTransferFunction('u')

    # get 2D transfer function for 'u'
    lTF2D = GetTransferFunction2D('u')

    # get layout
    layout1 = GetLayoutByName("Layout #1")

    # time of 0 to max, e��?e����: 5 s
    timeKeeper = GetTimeKeeper()
    time_values = timeKeeper.TimestepValues
    max_time = max(time_values)

    # create a new 'Plot Over Line' for -25, -15, -5, 5, 15, 25
    xlist = [-25, -15, -5, 5, 15, 25]
    for t in np.arange(0, max_time + 1e-6, 10.0):
        animationScene1.AnimationTime = t
        t_tag = f"t{int(t)}"
        screen_name = f"{base_name}_{t_tag}"
        export_screen_path = os.path.join(folder_path, screen_name + ".png").replace("\\", "/")
        SaveScreenshot(export_screen_path, view=renderView1)
        for x in xlist:
            plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=reader)

            # Properties modified on plotOverLine1
            plotOverLine1.Point1 = [x, 0.0, 0.0]
            plotOverLine1.Point2 = [x, 2.5, 0.0]

            # set active view
            SetActiveView(renderView1)

            # split cell
            layout1.SplitHorizontal(0, 0.5)

            # set active view
            SetActiveView(None)

            # Create a new 'SpreadSheet View'
            spreadSheetView1 = CreateView('SpreadSheetView')
            spreadSheetView1.ColumnToSort = ''
            spreadSheetView1.BlockSize = 1024

            # show data in view
            plotOverLine1Display_1 = Show(plotOverLine1, spreadSheetView1, 'SpreadSheetRepresentation')

            # Properties modified on spreadSheetView1
            # Properties modified on spreadSheetView1
            spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Points', 'Points_Magnitude',
                                                   'u_Magnitude', 'vtkValidPointMask']

            # set active source
            SetActiveSource(plotOverLine1)

            # export view
            t_tag = f"t{int(t)}"
            x_tag = f"x{int(x)}"
            csv_name = f"{base_name}_{t_tag}{x_tag}.csv"
            export_csv_path = os.path.join(folder_path, csv_name + ".csv").replace("\\", "/")
            ExportView(export_csv_path, view=spreadSheetView1)

            # destroy spreadSheetView1
            Delete(plotOverLine1)
            del plotOverLine1
            Delete(spreadSheetView1)
            del spreadSheetView1

            # layout
            layout1.Collapse(2)

    # destroy reader
    Delete(reader)
    del reader

    # destroy renderView1
    Delete(renderView1)
    del renderView1

##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------