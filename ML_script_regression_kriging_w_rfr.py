import sys

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from pykrige.rk import RegressionKriging

svr_model = SVR(C=0.15, gamma="auto")
rf_model = RandomForestRegressor(n_estimators=500)
lr_model = LinearRegression(copy_X=True, fit_intercept=False)

models = [svr_model, rf_model, lr_model]

############## my part ##################

import numpy as np
import pandas as pd
import geopandas as gpd
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


# Create empty lists to store coordinates
x_coords = []
y_coords = []

# Iterate over features
for feature in layer_dataset.getFeatures():
    # Get the geometry (point)
    geom = feature.geometry()
    # Extract coordinates
    x = geom.asPoint().x()
    y = geom.asPoint().y()
    # Append coordinates to lists
    x_coords.append(x)
    y_coords.append(y)



features_dataset = ['latitude', 'longitude', 'ISOPleasan', 'PARKSDISTAN', 'fountainsvi', 'roadnoiseco', 'treesvisibi', 'DISTANCE', 'VISIBLE']
features_dataset = ['latitude', 'longitude', 'ISOPleasan', 'PARKSDISTAN', 'roadnoiseco']

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


# Create a DataFrame with the feature values
if len(feature_values) > 0 and all(feature_values):
    train_data = pd.DataFrame(np.array(feature_values).T, columns=features_dataset)
else:
    print("No data available to create DataFrame.")

# Add coordinates to train_data DataFrame
train_data['X_Coordinates'] = x_coords
train_data['Y_Coordinates'] = y_coords
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
# train_data.drop(columns=['latitude', 'longitude'], inplace=True)

# Now 'train_data' contains the DataFrame with outliers in the 'ISOPleasan' column removed using the Z-score method
# print(train_data.head())
print('deleted', len(train_data)- old_length, 'from', old_length)

################### end my part ###############

"""
## fit for pykrige needs
p: predictor variables,
x: ndarray of points corresponding to the lon lat, 
y: array of targets

"""


#### fix the x input to 2d numpy array
train_data_x = train_data[['X_Coordinates', 'Y_Coordinates']].values.tolist()
train_data_x = np.array(train_data_x)
# print("column latlong ",train_data_x[0])
# print("type array", type(train_data_x), train_data_x.shape[1])


# Define the features sets
feature_sets = [
    ["PARKSDISTAN", "roadnoiseco"],
    ["PARKSDISTAN"],
    ["roadnoiseco"]
]

# Define the file path
file_path = "C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/rk_m_rf_04.csv"
# Specify the file path where you want to save the model
model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rk_rf04_regressor_parks_roadnoise.joblib'

# Create an empty DataFrame to store scores
scores_df = pd.DataFrame(columns=["Features", "Regression Score", "RK Score"])
scores = []
models_m_rk_rfr = []
# Loop through feature sets
for i, features in enumerate(feature_sets):
    print(features)
    train_data_p = train_data[features]
    train_data_p = np.array(train_data_p)
    
    p = train_data_p
    x = train_data_x
    target = train_data["ISOPleasan"]
    print(x[0])
    p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
        p, x, target, test_size=0.1, random_state=42
    )

    # Train the model

    m_rk_rfr = RegressionKriging(regression_model=rf_model, n_closest_points=20)
    m_rk_rfr.fit(p_train, x_train, target_train)

    # Calculate scores
    regression_score = m_rk_rfr.regression_model.score(p_test, target_test)
    rk_score = m_rk_rfr.score(p_test, x_test, target_test)

    # Append scores to list
    scores.append( [features, regression_score, rk_score] )
    models_m_rk_rfr.append(m_rk_rfr)
    # print('succesfull loop', i, scores)
    # scores = scores.append({"Features": f"Set {i+1}", "Regression Score": regression_score, "RK Score": rk_score}, ignore_index=True)

print(type(scores))

# scores_df = pd.concat(scores)
scores_df = (pd.DataFrame(data=scores, columns = ["Features", "Regression Score", "RK Score"]))
print(scores_df)

# # Save DataFrame as CSV
# scores_df.to_csv(file_path, index=False)
# # save regression kriging model
# # Save the model to the specified file path
# dump(models_m_rk_rfr[0], model_filename)







##### OLD

# train_data_p = train_data[["PARKSDISTAN", "roadnoiseco"]]
# train_data_p = train_data[["PARKSDISTAN"]]
# train_data_p = train_data[["roadnoiseco"]]
# train_data_p = np.array(train_data_p)

# p = train_data_p
# x = train_data_x
# target = train_data["ISOPleasan"]

# print('p', p)
# # print('x', x[0:2])
# # print('target', target)

# p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
#     p, x, target, test_size=0.1, random_state=42
# )


# print("=" * 40)
# print("regression model:", rf_model.__class__.__name__)
# m_rk_rfr = RegressionKriging(regression_model=rf_model, n_closest_points=10)
# m_rk_rfr.fit(p_train, x_train, target_train)
# print("Regression Score: ", m_rk_rfr.regression_model.score(p_test, target_test))
# print("RK score: ", m_rk_rfr.score(p_test, x_test, target_test))

# # Calculate scores
# regression_score = m_rk_rfr.regression_model.score(p_test, target_test)
# rk_score = m_rk_rfr.score(p_test, x_test, target_test)

# # Create a dictionary to store scores
# scores_dict = {
#     "Regression Score": regression_score,
#     "RK Score": rk_score
# }

# # Convert the dictionary to a DataFrame
# scores_df = pd.DataFrame([scores_dict])

# # Define the file path
# file_path = "C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/rk_m_rfr_01.csv"

# Save DataFrame as CSV
# scores_df.to_csv(file_path, index=False)

# print("File saved successfully at:", file_path)
# model_filename = 'C:/Users/n.smit/OneDrive - Oosterhoff Group/Documents/Master Thesis/Coding/predictions_QGIS/model_rf_regressor_only_parks.joblib'
# for m in models:
#     print("=" * 40)
#     print("regression model:", m.__class__.__name__)
#     m_rk = RegressionKriging(regression_model=m, n_closest_points=10)
#     m_rk.fit(p_train, x_train, target_train)
#     print("Regression Score: ", m_rk.regression_model.score(p_test, target_test))
#     print("RK score: ", m_rk.score(p_test, x_test, target_test))
    
