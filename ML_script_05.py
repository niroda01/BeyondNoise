import numpy as np
import pandas as pd
from qgis.core import QgsVectorLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from osgeo import gdal, osr
from qgis.core import QgsProject, QgsRasterLayer

from sklearn.ensemble import RandomForestRegressor
from joblib import dump


# Load the SSID DATASET point layer
layer_dataset = QgsVectorLayer("C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/QGIS/SSID_points_data/SSID_london_dataforml01.gpkg", "SSID_london_dataforml01 — ssid_dataforml01", "ogr")
print(layer_dataset)

# Check if the layer is valid
if not layer_dataset.isValid():
    print("Layer failed to load!")


features_dataset = ['latitude', 'longitude', 'ISOPleasan', 'PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'DISTANCE', 'VISIBLE']
features_dataset = ['latitude', 'longitude', 'ISOPleasan', 'PARKSDISTAN']

# Extract feature values from the point layer
feature_values = []
for feature in features_dataset:
    index = layer_dataset.fields().indexFromName(feature)
    if index != -1:  # Check if field exists
        try:
            values = [feature.attribute(index) for feature in layer_dataset.getFeatures()]
            values = [-99 if str(v) == 'NULL' else float(v) for v in values]
            feature_values.append(values)
        except Exception as e:
            print(f"Error extracting feature {feature}: {e}")
    else:
        print(f"Field {feature} not found.")

# # Print feature values to debug
# print("Feature Values:")
# for feature, values in zip(features_dataset, feature_values):
#     print(f"{feature}: {values}")

# Create a DataFrame with the feature values
if len(feature_values) > 0 and all(feature_values):
    train_data = pd.DataFrame(np.array(feature_values).T, columns=features_dataset)
else:
    print("No data available to create DataFrame.")

old_length = len(train_data)

# remove weird outliers

# Step 1: Group by identical latitude and longitude, and count the number of points in each group
grouped_stats = train_data.groupby(['latitude', 'longitude']).size().reset_index(name='row_count')

# Step 2: Filter places with 5 or more points
places_with_5_or_more_points = grouped_stats[grouped_stats['row_count'] >= 3]

# Step 3: Iterate over each group and remove outliers in the 'ISOPleasan' column using the Z-score method
for index, group in places_with_5_or_more_points.iterrows():
    # Filter data for the current group
    group_data = train_data[(train_data['latitude'] == group['latitude']) &
                            (train_data['longitude'] == group['longitude'])]

    # Calculate z-scores for the 'ISOPleasan' column
    z_scores = np.abs((group_data['ISOPleasan'] - group_data['ISOPleasan'].mean()) / group_data['ISOPleasan'].std())

    # Define the threshold for outlier detection (z-score > 1)
    outlier_threshold = 2

    # Filter outliers based on the defined threshold
    outliers_to_remove = group_data[z_scores > outlier_threshold]

    # Remove outliers from the original DataFrame
    train_data = train_data.drop(outliers_to_remove.index)

train_data = train_data.dropna()
train_data.drop(columns=['latitude', 'longitude'], inplace=True)

# Now 'train_data' contains the DataFrame with outliers in the 'ISOPleasan' column removed using the Z-score method
# print(train_data.head())
print('deleted', len(train_data)- old_length, 'from', old_length)

# Splitting the dataset into training and testing sets:  train_set contains 75% of the data, test_set contains 25% of the data
train_set, test_set = train_test_split(train_data, test_size=0.10, random_state=42)
# print("Training set shape:", train_set.shape, "Testing set shape:", test_set.shape)

# Creating X_train, y_train, X_test, y_test
x_train = train_set.drop(columns=['ISOPleasan'])
y_train = train_set['ISOPleasan']
x_test = test_set.drop(columns=['ISOPleasan'])
y_test = test_set['ISOPleasan']

# Initializing the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=5000, random_state=42, oob_score=True, max_depth = 4, min_samples_leaf = 3) #, max_features = 'sqrt')

# Training the model
rf_regressor.fit(x_train, y_train)

# Predicting on the test set
y_pred = rf_regressor.predict(x_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("random forest regressor results")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
# Access the OOB Score
oob_score = rf_regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')




# Specify the file path where you want to save the model
model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rf_regressor_only_parks.joblib'

# Save the model to the specified file path
# dump(rf_regressor, model_filename)


