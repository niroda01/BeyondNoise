from qgis.core import QgsProject, QgsField, QgsVectorLayer, QgsFeature
from PyQt5.QtCore import QVariant
import processing
import pandas as pd

# List of raster layers to analyze
raster_layers = [
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/00 without allterations/prediction_raster_existing.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/00 without allterations/prediction_raster_add_fountains07.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/00 without allterations/prediction_raster_alteration2_added park.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/00 without allterations/prediction_raster_alteration2_added park_and_fountains07.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign06.sdat', 
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign06_parks07.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign06_fountains07_parks07.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign07.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign07_parks.sdat',
    'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/newsite06/prediction_raster_redesign07_fountains_park.sdat'
    
]

# List of custom names for the output layers
layer_names = [
    'b0p0f0', 
    'b0p0f1',
    'b0p1f0',
    'b0p1f1',
    'b1p0f0',
    'b1p1f0',
    'b1p1f1',
    'b2p0f0',
    'b2p1f0',
    'b2p1f1'
    
]

# Path to the vector layer
vector_layer = 'C:\\Users\\n.smit\\OneDrive - Oosterhoff Group\\Documents\\Master Thesis\\QGIS\\newsite01\\newsite06\\zonal_stats_parks01.shp'

# List to store DataFrame for each output layer
dataframes = []

# Iterate over each raster layer and custom name
for raster_layer, name in zip(raster_layers, layer_names):
    # Run the Zonal Statistics tool
    result = processing.run("native:zonalstatisticsfb", {
        'INPUT': vector_layer,
        'INPUT_RASTER': raster_layer,
        'RASTER_BAND': 1,
        'COLUMN_PREFIX': '_',
        'STATISTICS': [2, 3, 4, 5, 6, 7],
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })
    
    # Get the output layer from the result
    output_layer = result['OUTPUT']
    
    # Add a new field 'type' to the output layer
    type_field = QgsField('type', QVariant.String)
    output_layer.addAttribute(type_field)
    
    # Start editing the output layer
    output_layer.startEditing()
    
    # Update feature attributes with the layer name in the 'type' field
    for feature in output_layer.getFeatures():
        output_layer.changeAttributeValue(feature.id(), output_layer.fields().indexFromName('type'), name)
    
    # Commit changes and stop editing
    output_layer.commitChanges()
    
    # Set the custom name for the output layer
    output_layer.setName(name)

    # Add the output layer to the map
    # QgsProject.instance().addMapLayer(output_layer)
    
    # Convert the output layer to a pandas DataFrame
    field_names = [field.name() for field in output_layer.fields()]
    features = output_layer.getFeatures()
    attribute_data = [[feature[field_name] for field_name in field_names] for feature in features]
    df = pd.DataFrame(attribute_data, columns=field_names)
    
    # Add a column for the layer name
    df['type'] = name
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Specify the columns you want to round
columns_to_round = ['_mean', '_median', '_stdev', '_min', '_max', '_range']

# Round the values in the specified columns
combined_df[columns_to_round] = combined_df[columns_to_round].round(2)

# Print the DataFrame to verify the changes
print(combined_df)



# Save the combined DataFrame to a CSV file
csv_file_path = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/predictions.csv'  # Specify the path where you want to save the CSV file
# combined_df.to_csv(csv_file_path, index=True) 

# processing.run("native:zonalstatisticsfb", {'INPUT':'C:\\Users\\n.smit\\OneDrive - Oosterhoff Group\\Documents\\Master Thesis\\QGIS\\newsite01\\newsite06\\zonal_stats_parks01.shp','INPUT_RASTER':'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/newsite01/00 without allterations/prediction_raster_existing.sdat','RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[0,1,2],'OUTPUT':'TEMPORARY_OUTPUT'})