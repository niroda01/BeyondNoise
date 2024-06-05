"""
Model exported as python.
Name : model
Group : 
With QGIS : 33600
"""

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterRasterDestination
import processing
from qgis.PyQt.QtCore import QCoreApplication

class Create_parks_dist(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer('parks', 'PARKS', types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterDestination('Parks_dist', 'PARKS_DIST', createByDefault=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(5, model_feedback)
        results = {}
        outputs = {}

        # Parks Rasteriseren (vector naar raster)
        alg_params = {
            'BURN': 1,
            'DATA_TYPE': 5,  # Float32
            'EXTENT': parameters['parks'],
            'EXTRA': '',
            'FIELD': '',
            'HEIGHT': 1,
            'INIT': None,
            'INPUT': parameters['parks'],
            'INVERT': False,
            'NODATA': 0,
            'OPTIONS': '',
            'UNITS': 1,  # Eenheden voor geoverwijzingen
            'USE_Z': False,
            'WIDTH': 1,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['ParksRasteriserenVectorNaarRaster'] = processing.run('gdal:rasterize', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Proximity OUTSIDE (raster afstand)
        alg_params = {
            'BAND': 1,
            'DATA_TYPE': 5,  # Float32
            'EXTRA': '',
            'INPUT': outputs['ParksRasteriserenVectorNaarRaster']['OUTPUT'],
            'MAX_DISTANCE': 200,
            'NODATA': 200,
            'OPTIONS': '',
            'REPLACE': 0,
            'UNITS': 0,  # Coördinaten met geoverwijzingen
            'VALUES': '',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['ProximityOutsideRasterAfstand'] = processing.run('gdal:proximity', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Parks inverse Rasteriseren (vector naar raster)
        alg_params = {
            'BURN': 1,
            'DATA_TYPE': 5,  # Float32
            'EXTENT': parameters['parks'],
            'EXTRA': '',
            'FIELD': '',
            'HEIGHT': 1,
            'INIT': None,
            'INPUT': parameters['parks'],
            'INVERT': True,
            'NODATA': 0,
            'OPTIONS': '',
            'UNITS': 1,  # Eenheden voor geoverwijzingen
            'USE_Z': False,
            'WIDTH': 1,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['ParksInverseRasteriserenVectorNaarRaster'] = processing.run('gdal:rasterize', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        # Proximity INSIDE (raster afstand)
        alg_params = {
            'BAND': 1,
            'DATA_TYPE': 5,  # Float32
            'EXTRA': '',
            'INPUT': outputs['ParksInverseRasteriserenVectorNaarRaster']['OUTPUT'],
            'MAX_DISTANCE': 200,
            'NODATA': 200,
            'OPTIONS': '',
            'REPLACE': 0,
            'UNITS': 0,  # Coördinaten met geoverwijzingen
            'VALUES': '',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['ProximityInsideRasterAfstand'] = processing.run('gdal:proximity', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        # PARKS Raster difference
        alg_params = {
            'A': outputs['ProximityInsideRasterAfstand']['OUTPUT'],
            'B': outputs['ProximityOutsideRasterAfstand']['OUTPUT'],
            'C': parameters['Parks_dist']
        }
        outputs['ParksRasterDifference'] = processing.run('sagang:rasterdifference', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Parks_dist'] = outputs['ParksRasterDifference']['C']
        return results


    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)
        
    def name(self):
        return 'Create_parks_dist'

    def displayName(self):
        return 'Create_parks_dist'

    def group(self):
        return self.tr('Machine Learning')

    def groupId(self):
        return 'machine_learning'

    def createInstance(self):
        return Create_parks_dist()
