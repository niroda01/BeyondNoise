"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)
from qgis import processing
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point
from pyproj import CRS
from qgis.core import QgsVectorLayer, QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource, QgsProcessingParameterFeatureSink, QgsField, QgsWkbTypes, QgsPointXY, QgsProcessing
from sklearn.ensemble import RandomForestRegressor
import joblib

class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExampleProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'points_predict_values'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Predict values for points')

    def group(self):
        return self.tr('Machine Learning')

    def groupId(self):
        return 'machine_learning'


    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Example algorithm short description")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.SourceType.TypeVectorAnyGeometry]
            )
        )

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )



    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        source = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )
        layer_points = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        
        columns_points = ['PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'vi20mbigger', 'averageview', 'skyviewfact', 'visibleskyl']
        columns = ['PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'vi20mbigger', 'DISTANCE', 'SVF', 'VISIBLE']
        
                # Extract feature values from the point layer
        feature_values = []
        for feature in columns_points:
            index = layer_points.fields().indexFromName(feature)
            if index != -1:  # Check if field exists
                try:
                    values = [feature.attribute(index) for feature in layer_points.getFeatures()]
                    values = [-99 if str(v) == 'NULL' else float(v) for v in values]
                    feature_values.append(values)
                except Exception as e:
                    feedback.reportError(f"Error extracting feature {feature}: {e}")
            else:
                feedback.reportError(f"Field {feature} not found.")

        if len(feature_values) > 0 and all(feature_values):
            points_data_for_prediction = pd.DataFrame(np.array(feature_values).T, columns=columns)
        else:
            feedback.reportError("No data available to create DataFrame.")
            return {}

        # Load the trained Random Forest regressor model
        model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rf_regressor06.joblib'
        rf_regressor = joblib.load(model_filename)

        # Predict using the Random Forest regressor
        points_predicted_ISOPleasan = rf_regressor.predict(points_data_for_prediction)

        # Add predictions to the DataFrame
        points_data_for_prediction['Predicted_ISOPleasan'] = points_predicted_ISOPleasan

        # Extract point coordinates from the layer
        x_coordinates = []
        y_coordinates = []
        for feature in layer_points.getFeatures():
            geometry = feature.geometry()
            if geometry.type() == QgsWkbTypes.PointGeometry:
                x = geometry.asPoint().x()
                y = geometry.asPoint().y()
                x_coordinates.append(x)
                y_coordinates.append(y)
            else:
                feedback.reportError("Geometry is not a point")

        points_data_for_prediction['x'] = x_coordinates
        points_data_for_prediction['y'] = y_coordinates

        # combine x and y column to a shapely Point() object
        points_data_for_prediction['geometry'] = points_data_for_prediction.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)

        # Create a GeoDataFrame with specified CRS
        gdf_points_with_prediction = geopandas.GeoDataFrame(points_data_for_prediction, geometry='geometry', crs=CRS.from_epsg(3857))

        # Save the GeoDataFrame to a file
        filepath_output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        gdf_points_with_prediction.to_file(filepath_output, driver='GPKG')

        return {self.OUTPUT: filepath_output}
        
        # # If source was not found, throw an exception to indicate that the algorithm
        # # encountered a fatal error. The exception text can be any string, but in this
        # # case we use the pre-built invalidSourceError method to return a standard
        # # helper text for when a source cannot be evaluated
        # if source is None:
        #     raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        # (sink, dest_id) = self.parameterAsSink(
        #     parameters,
        #     self.OUTPUT,
        #     context,
        #     source.fields(),
        #     source.wkbType(),
        #     source.sourceCrs()
        # )

        # # Send some information to the user
        # feedback.pushInfo(f'CRS is {source.sourceCrs().authid()}')

        # # If sink was not created, throw an exception to indicate that the algorithm
        # # encountered a fatal error. The exception text can be any string, but in this
        # # case we use the pre-built invalidSinkError method to return a standard
        # # helper text for when a sink cannot be evaluated
        # if sink is None:
        #     raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        # # Compute the number of steps to display within the progress bar and
        # # get features from source
        # total = 100.0 / source.featureCount() if source.featureCount() else 0
        # features = source.getFeatures()

        # for current, feature in enumerate(features):
        #     # Stop the algorithm if cancel button has been clicked
        #     if feedback.isCanceled():
        #         break

        #     # Add a feature in the sink
        #     sink.addFeature(feature, QgsFeatureSink.Flag.FastInsert)

        #     # Update the progress bar
        #     feedback.setProgress(int(current * total))

        # # To run another Processing algorithm as part of this algorithm, you can use
        # # processing.run(...). Make sure you pass the current context and feedback
        # # to processing.run to ensure that all temporary layer outputs are available
        # # to the executed algorithm, and that the executed algorithm can send feedback
        # # reports to the user (and correctly handle cancellation and progress reports!)
        # if False:
        #     buffered_layer = processing.run("native:buffer", {
        #         'INPUT': dest_id,
        #         'DISTANCE': 1.5,
        #         'SEGMENTS': 5,
        #         'END_CAP_STYLE': 0,
        #         'JOIN_STYLE': 0,
        #         'MITER_LIMIT': 2,
        #         'DISSOLVE': False,
        #         'OUTPUT': 'memory:'
        #     }, context=context, feedback=feedback)['OUTPUT']

        # # Return the results of the algorithm. In this case our only result is
        # # the feature sink which contains the processed features, but some
        # # algorithms may return multiple feature sinks, calculated numeric
        # # statistics, etc. These should all be included in the returned
        # # dictionary, with keys matching the feature corresponding parameter
        # # or output names.
        # return {self.OUTPUT: dest_id}
