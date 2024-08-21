# Install necessary packages if not already installed
!pip install folium
!pip install pandas

import folium
import pandas as pd
import requests
import io

# Import folium plugins
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon

# Download and read the `spacex_launch_geo.csv` using requests
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
response = requests.get(URL)
spacex_df = pd.read_csv(io.StringIO(response.text))

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df

# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

# Create a circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a circle at NASA Johnson Space Center's coordinate with an icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)

# Initialize the map
default_location = [nasa_coordinate[0] + 10, nasa_coordinate[1]]
site_map = folium.Map(location=default_location, zoom_start=5.2)

# Iterate through each launch site
for index, row in launch_sites_df.iterrows():
    # Create a circle at launch site location with a popup label showing its name
    circle = folium.Circle(
        location=[row['Lat'], row['Long']],
        radius=1000,
        color='red',
        fill=True,
        fill_color='#d35400',  # Transparent orange fill
        fill_opacity=0.6,
    ).add_to(site_map)

    # Add custom text label to the circle
    text_label = folium.Marker(
        location=[row['Lat'], row['Long']],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size: 12px; color: red;">%s</div>' % row['Launch Site']
        )
    ).add_to(site_map)

# Display the map
site_map

# Marker cluster setup
marker_cluster = MarkerCluster()

# Function to map class values to marker colors
def assign_marker_color(row):
    if row == 1:
        return 'green'
    elif row == 0:
        return 'red'
    else:
        return 'black' 

# Apply the function to create the 'marker_color' column in spacex_df
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

# Add marker_cluster to current site_map
for index, row in spacex_df.iterrows():
    marker = folium.Marker(
        location=[row['Lat'], row['Long']],
        icon=folium.Icon(color='white', icon_color=row['marker_color'])
    )
    marker_cluster.add_child(marker)

# Add clusters to map
site_map.add_child(marker_cluster)

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)

# Display the updated map
site_map

# Distance calculation function
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Find the coordinate of the closest coastline
launch_site_lat = 28.573255
launch_site_lon = -80.646895
coastline_lat = 28.59402
coastline_lon = -80.58163

# Calculate the distance between the launch site and the coastline
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
print("Distance to closest coastline:", distance_coastline, "km")

# Create and add a folium.Marker on your selected closest coastline point on the map
distance_marker = folium.Marker(
    [coastline_lat, coastline_lon],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
    )
)
site_map.add_child(distance_marker)

# Create a PolyLine object
coordinates = [[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]]
line = folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(line)

# Display the updated map
site_map

# Distance to closest city
closest_city_lat = 28.612219
closest_city_lon = -80.8077556

# Calculate the distance between the launch site and the closest city
distance_city = calculate_distance(launch_site_lat, launch_site_lon, closest_city_lat, closest_city_lon)
print("Distance to closest city:", distance_city, "km")

# Create a marker for the closest city point
city_marker = folium.Marker(
    [closest_city_lat, closest_city_lon],
    icon=folium.Icon(color='orange', icon='info-sign')
)
site_map.add_child(city_marker)

# Display the distance between the city point and launch site using the icon property
distance_marker_city = folium.Marker(
    [closest_city_lat, closest_city_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_city),
    )
)
site_map.add_child(distance_marker_city)

# Draw a PolyLine between the closest city and the launch site
coordinates_city = [[launch_site_lat, launch_site_lon], [closest_city_lat, closest_city_lon]]
line_city = folium.PolyLine(locations=coordinates_city, weight=1)
site_map.add_child(line_city)

# Distance to closest railway
closest_railway_lat = 28.58058  
closest_railway_lon = -80.59098  

# Calculate the distance between the launch site and the closest railway
distance_railway = calculate_distance(launch_site_lat, launch_site_lon, closest_railway_lat, closest_railway_lon)
print("Distance to closest railway:", distance_railway, "km")

# Create a marker for the closest railway point
railway_marker = folium.Marker(
    [closest_railway_lat, closest_railway_lon],
    icon=folium.Icon(color='blue', icon='train')
)
site_map.add_child(railway_marker)

# Display the distance between the railway point and launch site using the icon property
distance_marker_railway = folium.Marker(
    [closest_railway_lat, closest_railway_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_railway),
    )
)
site_map.add_child(distance_marker_railway)

# Draw a PolyLine between the closest railway and the launch site
coordinates_railway = [[launch_site_lat, launch_site_lon], [closest_railway_lat, closest_railway_lon]]
line_railway = folium.PolyLine(locations=coordinates_railway, weight=1)
site_map.add_child(line_railway)

# Distance to closest highway
closest_highway_lat = 28.5728
closest_highway_lon = -80.65561

# Calculate the distance between the launch site and the closest highway
distance_highway = calculate_distance(launch_site_lat, launch_site_lon, closest_highway_lat, closest_highway_lon)
print("Distance to closest highway:", distance_highway, "km")

# Create a marker for the closest highway point
highway_marker = folium.Marker(
    [closest_highway_lat, closest_highway_lon],
    icon=folium.Icon(color='orange', icon='road')
)
site_map.add_child(highway_marker)

# Display the distance between the highway point and launch site using the icon property
distance_marker_highway = folium.Marker(
    [closest_highway_lat, closest_highway_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_highway),
    )
)
site_map.add_child(distance_marker_highway)

# Draw a PolyLine between the closest highway and the launch site
coordinates_highway = [[launch_site_lat, launch_site_lon], [closest_highway_lat, closest_highway_lon]]
line_highway = folium.PolyLine(locations=coordinates_highway, weight=1)
site_map.add_child(line_highway)

# Display the updated map
site_map
