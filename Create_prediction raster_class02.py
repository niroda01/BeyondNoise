"""
Model exported as python.
Name : model for prediction
Group : 
With QGIS : 33600
"""

from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterNumber
from qgis.core import QgsProcessingParameterExtent
from qgis.core import QgsProcessingParameterVectorDestination
import processing

import lightgbm as lgb
import numpy as np
import pandas as pd
from qgis.core import QgsVectorLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from osgeo import gdal, osr
from qgis.core import QgsProject, QgsRasterLayer


import numpy as np
import pandas as pd
from qgis.core import QgsProcessingAlgorithm, QgsProcessingParameterRasterDestination, QgsProcessingParameterFeatureSource, QgsProcessingOutputRasterLayer
from qgis.core import QgsRasterLayer
from osgeo import gdal, osr
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib


class PredictRasterAlgorithm(QgsProcessingAlgorithm):
    # MODEL_FILE = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite02/prediction/mode.txt'
    # Load the trained Random Forest regressor model
    
    # rf_regressor = joblib.load(model_filename)
    # MODEL_FILE = rf_regressor
    PIXEL_SIZE = 3
    EPSG = "3857"
    
    def initAlgorithm(self, config=None):
        # self.addParameter(QgsProcessingParameterVectorLayer('GRIDPOINTS', 'GRIDPOINTS DESIGN', types=[QgsProcessing.TypeVectorPoint], defaultValue=None))
        self.addParameter(QgsProcessingParameterNumber('grid_size_meters', 'grid size (meters)', type=QgsProcessingParameterNumber.Double, defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSource('GRIDPOINTS_LAYER', 'Grid Points Layer'))
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                'OUTPUT_RASTER',
                'Output Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        # Load the model
        model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rf_regressor06.joblib'
        model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rf_regressor_parks_lden.joblib'
        rf_regressor = joblib.load(model_filename)
        model = rf_regressor
        
        # Load the grid points layer
        grid_points_layer = self.parameterAsVectorLayer(parameters, 'GRIDPOINTS_LAYER', context)
        
        # Extract feature values from the point layer
        features_all = ['left', 'top', 'right', 'bottom', 'row_index', 'col_index', 'Parksdist', 'RESULT','OUTPUT','OUTPUT_1', 'DISTANCE', 'VISIBLE']
        # columns_all = ['left', 'top', 'right', 'bottom', 'row_index', 'col_index','vi100m', 'vi20m', 'viewshed_t', 'parks_rast', 'parks_dist', 'fountainsvi', 'road_noise']
        columns_all = ['left', 'top', 'right', 'bottom', 'row_index', 'col_index', 'PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'DISTANCE', 'VISIBLE']
        columns = ['PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'DISTANCE', 'VISIBLE']
        
        feature_values = []
        
        for feature in features_all:
            print("Feature:", feature)
            print("Number of fields:", len(grid_points_layer.fields()))
            index = grid_points_layer.fields().indexFromName(feature)
            print("Index:", index)
            values = [feature.attribute(index) for feature in grid_points_layer.getFeatures()]
            values = [-1 if str(v) == 'NULL' else float(v) for v in values]
            feature_values.append(values)

        # Create a DataFrame with the feature values
        grid_points_data = pd.DataFrame(np.array(feature_values).T, columns=columns_all)
        
        # Create df for the input of the prediction model
        grid_points_data_input = grid_points_data[columns]
        
        # Make predictions
        grid_points_predictions = model.predict(grid_points_data_input)
        
        # Add predicted values as a new column
        grid_points_data['Predicted_ISOPleasan'] = grid_points_predictions
    
        # Create columns for the lat & lon (used for the raster creation)
        # grid_points_data['lon'] = (grid_points_data['left'] + grid_points_data['right']) / 2
        grid_points_data['lon'] = grid_points_data['left'] - abs((grid_points_data['left'] - grid_points_data['right']) / 2)
        grid_points_data['lat'] = (grid_points_data['top'] + grid_points_data['bottom']) / 2
                
        # Set up raster parameters
        xmin, ymin, xmax, ymax = (grid_points_data['lon'].min(), grid_points_data['lat'].min(), 
                           grid_points_data['lon'].max(), grid_points_data['lat'].max())
        n_cols = int(grid_points_data['col_index'].max() + 1)
        n_rows = int(grid_points_data['row_index'].max() + 1)
        
        # Path to the raster file
        raster_file = self.parameterAsOutputLayer(parameters, 'OUTPUT_RASTER', context)
        
        # Create an empty raster
        target_ds = gdal.GetDriverByName('GTiff').Create(raster_file, n_cols, n_rows, 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform((xmin, parameters['grid_size_meters'], 0, ymax, 0, -parameters['grid_size_meters']))
        
        # Create a spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(self.EPSG))
        target_ds.SetProjection(srs.ExportToWkt())
        
        # Create a numpy array to hold the raster values
        raster_data = np.zeros((n_rows, n_cols), dtype=np.float32)
        
        # Populate raster values from the DataFrame
        for index, row in grid_points_data.iterrows():
            col_val = int((row['left'] - xmin) / parameters['grid_size_meters'])
            row_val = int((ymax - row['top']) / parameters['grid_size_meters'])
            # print(col_val, row_val)
            raster_data[row_val, col_val] = row['Predicted_ISOPleasan']
        
        # Write the raster data to the raster band
        band = target_ds.GetRasterBand(1)
        band.WriteArray(raster_data)
        
        # Close the raster dataset
        target_ds = None
        
        return {'OUTPUT_RASTER': raster_file}

    def name(self):
        return 'predict_raster02'

    def displayName(self):
        return 'Predict Raster02'

    def group(self):
        return 'Example Scripts'

    def groupId(self):
        return 'examplescripts'

    def createInstance(self):
        return PredictRasterAlgorithm()
