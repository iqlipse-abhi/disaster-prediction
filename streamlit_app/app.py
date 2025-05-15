import streamlit as st 
import pandas as pd
import requests
from joblib import load
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET
from geonamescache import GeonamesCache
import pycountry
import os

HAZARD_WEIGHTS = {
    "earthquake": {"base": 2.5},
    "flood":      {"base": 1.8},
    "cyclone":    {"base": 2.2},
    "volcano":    {"base": 2.6},
    "wildfire":   {"base": 2.0},
    "drought":    {"base": 1.4},
    "other":      {"base": 1.2},
}

SEVERITY_FACTORS = {"green": 1.0, "orange": 1.25, "red": 1.5}

dataset_df = pd.read_csv('final_features_risk_calculator.csv')
reference_dataset = pd.read_csv('final_dataset_risk_calculator.csv')
model = load('disaster_risk_calculator_model.joblib')

merged_df = dataset_df.copy()
merged_df[['Country','Subregion','Year','ISO3.Code']] = reference_dataset[['Country','Subregion','Year','ISO3.Code']]

le_subregion = LabelEncoder()
le_country = LabelEncoder()
le_subregion.fit(merged_df['Subregion'])
le_country.fit(merged_df['Country'])

st.markdown("## Disaster Risk Score Predictor")
location_method = st.radio("Choose how to provide your location:", ["Detect my location", "I'll enter manually"])

lat, lon = None, None
selected_country = ""
city_name = ""

if location_method == "Detect my location":
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        loc = ip_info.get("loc", "")
        if loc:
            lat, lon = map(float, loc.split(","))
            city_name = ip_info.get("city", "Unknown")
            iso2_code = ip_info.get("country", "Unknown")
            
            def iso2_to_country_name(code):
                try:
                    return pycountry.countries.get(alpha_2=code).name
                except:
                    return "Unknown"
            
            selected_country = iso2_to_country_name(iso2_code)
            st.success(f"Detected Location: {city_name}, {selected_country}")
        else:
            st.error("Couldn't detect your location.")
            st.stop()
    except Exception as e:
        st.error(f"Location detection error: {e}")
        st.stop()

    country_data = merged_df[merged_df['Country'] == selected_country].sort_values(by='Year', ascending=False)
    if country_data.empty:
        st.warning(f"No data found for {selected_country}. Please try entering it manually.")
        st.stop()

else:
    country_list = merged_df['Country'].dropna().unique()
    selected_country = st.selectbox("Select your Country", sorted(country_list))
    selected_iso3 = merged_df[merged_df['Country'] == selected_country]['ISO3.Code'].values[0]

    def iso3_to_iso2(iso3_code):
        try:
            return pycountry.countries.get(alpha_3=iso3_code).alpha_2
        except:
            return None
    
    iso2_code = iso3_to_iso2(selected_iso3)
    
    gc = GeonamesCache()
    all_cities = gc.get_cities()

    if iso2_code:
        filtered_city_names = [
            city['name'] for city in all_cities.values()
            if city['countrycode'] == iso2_code
        ]
        if filtered_city_names:
            selected_city = st.selectbox("Select your City", sorted(filtered_city_names))
        else:
            selected_city = st.text_input("Enter your City Name (no cities found for this country)")
    else:
        st.warning("Could not convert ISO3 code to ISO2. Enter city manually.")
        selected_city = st.text_input("Enter your City Name manually")

    city_name = selected_city
    country_data = merged_df[merged_df['Country'] == selected_country].sort_values(by='Year', ascending=False)

housing_material = st.selectbox("Select primary housing material:", ['Concrete / Brick', 'Wood', 'Tin', 'Mud', 'Straw / Leaves', 'Other'])

if selected_country and housing_material and ((location_method == "Detect my location" and city_name and lat and lon) or (location_method == "I'll enter manually" and selected_city)):
    country_data = merged_df[merged_df['Country'] == selected_country].sort_values(by='Year', ascending=False)

    slum_percent = country_data['Proportion of urban population living in slums or informal settlements (%) (a)'].dropna()
    slum_percent = slum_percent.iloc[0] / 100 if not slum_percent.empty else merged_df['Proportion of urban population living in slums or informal settlements (%) (a)'].mean() / 100

    material_weights = {
        'Mud': 1.3,
        'Tin': 1.2,
        'Wood': 1.1,
        'Concrete / Brick': 1.0,
        'Straw / Leaves': 1.4,
        'Other': 1.3
    }
    housing_multiplier = material_weights.get(housing_material, 1.0)
    housing_score = slum_percent * housing_multiplier

    historical_disaster_risk = 0
    
    latest_data_row = country_data.iloc[0]
    
    for col in ['E', 'V', 'S', 'C', 'A']:
        if col in latest_data_row:
            latest_data_row[col] = latest_data_row[col] / 10.0
        
    long_term_risk_score = (latest_data_row['E'] + latest_data_row['V'] + latest_data_row['S'] + latest_data_row['C'] + latest_data_row['A']) / 5

    def prepare_features(latest_row):
        latest_row = latest_row.copy()
        encoded_country = le_country.transform([latest_row['Country']])[0]
        encoded_subregion = le_subregion.transform([latest_row['Subregion']])[0]
        latest_row = latest_row.drop(['Country', 'Subregion', 'ISO3.Code', 'Year'], errors='ignore')
        latest_row['Country_encoded'] = encoded_country
        latest_row['Subregion_encoded'] = encoded_subregion
        return latest_row
    
    
    prepared_features = prepare_features(latest_data_row)
    model_score = model.predict([prepared_features])[0]
    
    def infer_disaster_type(text):
        text = text.lower()
        if 'earthquake' in text:
            return 'earthquake'
        elif 'flood' in text:
            return 'flood'
        elif 'cyclone' in text or 'storm' in text:
            return 'cyclone'
        elif 'volcano' in text:
            return 'volcano'
        elif 'wildfire' in text or 'fire' in text:
            return 'wildfire'
        elif 'drought' in text or 'dry storm' in text:
            return 'drought'
        else:
            return 'other'
    
    
    def infer_severity(description):
        description = description.lower()
        if 'red' in description:
            return 'red'
        elif 'orange' in description:
            return 'orange'
        elif 'green' in description:
            return 'green'
        else:
            return 'green'

    def check_gdacs_disaster(country_name):
        gdacs_url = "https://www.gdacs.org/xml/rss.xml"
        try:
            gdacs_resp = requests.get(gdacs_url)
            if gdacs_resp.status_code == 200:
                root = ET.fromstring(gdacs_resp.content)
                for item in root.iter('item'):
                    title = item.find('title').text
                    description = item.find('description').text
                    if country_name.lower() in title.lower() or country_name.lower() in description.lower():
                        disaster_type = infer_disaster_type(title)
                        severity = infer_severity(description)
                        return True, disaster_type, severity, title, description
            return False, None, None, None, None
        except Exception as e:
            st.warning(f"Could not fetch GDACS data: {e}")
            return False, None, None, None, None
    
    disaster_active, disaster_type, severity, disaster_title, disaster_description = check_gdacs_disaster(selected_country)
    
    weather_multiplier = 1.0
    weather_condition = "Unknown"
    
    if city_name:
        geo_url = "https://nominatim.openstreetmap.org/search"
        try:
            geo_resp = requests.get(geo_url, params={"q": f"{city_name}, {selected_country}", "format": "json"}, headers={"User-Agent": "DisasterRiskApp/1.0"})
            if geo_resp.status_code == 200:
                geo_data = geo_resp.json()
                if geo_data:
                    lat = float(geo_data[0]["lat"])
                    lon = float(geo_data[0]["lon"])
        except Exception as e:
            st.error(f"Geolocation error: {e}")
        
        
    if lat and lon:
        weather_api_key = os.getenv("WEATHER_API_KEY")
        try:
            weather_resp = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api_key}&units=metric")
            weather_data = weather_resp.json()
            if 'weather' in weather_data:
                weather_condition = weather_data['weather'][0]['main']
                weather_weights = {
                    'Rain': 1.1,
                    'Thunderstorm': 1.2,
                    'Extreme': 1.3,
                    'Snow': 1.1,
                    'Clear': 1.0,
                    'Clouds': 1.0
                }
                weather_multiplier = weather_weights.get(weather_condition, 1.0)
                temperature = weather_data['main']['temp']
                st.markdown("## 🌦️ Current Weather")
                with st.container():
                    st.write(f"**Location:** {city_name}, {selected_country}")
                    st.write(f"**Weather Condition:** {weather_condition}")
                    if 'main' in weather_data:
                        st.write(f"**Temperature:** {weather_data['main']['temp']} °C")
                        st.write(f"**Humidity:** {weather_data['main']['humidity']}%")
        except Exception as e:
            st.error(f"Weather data error: {e}")
    else:
        st.error("City coordinates could not be determined.")
        
        
    background_score = model_score * housing_score * weather_multiplier
    if disaster_active:
        base = HAZARD_WEIGHTS.get(disaster_type, HAZARD_WEIGHTS["other"])["base"]
        sev  = SEVERITY_FACTORS.get(severity, 1.0)
        
        disaster_score = (background_score * 0.5) + (base * sev * 2.0)
        
        st.markdown("## 🚨 Real-time Disaster Alert")
        with st.container():
            if disaster_active:
                st.error(f"**{disaster_title}**")
                st.write(f"**Severity Level:** {severity.capitalize()}")
                st.write(f"**Details:** {disaster_title}")
                st.info(disaster_description)
            else:
                st.success("No active disasters reported in your area.")
        
    else:
        disaster_score = background_score * 0.7
        
    disaster_score = round(disaster_score, 2)

    st.markdown("## 🧮 Final Disaster Risk Score")
    with st.container():
        if disaster_active:
            st.markdown("***Active disaster detected. Risk score includes real-time severity.***")
        else:
                st.markdown("***No active disaster. Score reflects long-term vulnerability only.***")
        st.write(f"Disaster Risk Score for {city_name}, {selected_country}: {disaster_score}")
    
    if disaster_active:
        severity_display = {
        "green": "Low",
        "orange": "Moderate",
        "red": "High"
    }
        if severity_display.get(severity,"Unknown") == "Low":
            st.success("Low Risk")
        elif severity_display.get(severity,"Unknown") == "Moderate":
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")
        
    else:
        if disaster_score < 1.5:
            level = "Low"
        elif disaster_score < 3.0:
            level = "Moderate"
        else:
            level = "High"
        st.write("Background Risk Level: ")
        if level == "Low":
            st.success("Low Risk")
        elif level == "Moderate":
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")


    prepared_features = prepare_features(latest_data_row)
    model_score = model.predict([prepared_features])[0]
    
    real_time_risk_score = (model_score * 0.2) + (weather_multiplier * 0.3) + (disaster_score * 0.5)
    real_time_risk_score = min(real_time_risk_score, 9.5)
    
    threshold = 2.0
    if real_time_risk_score > threshold:
        final_risk_score = round(0.3 * long_term_risk_score + 0.7 * real_time_risk_score, 2)
    else:
        final_risk_score = round(0.6 * long_term_risk_score + 0.4 * real_time_risk_score, 2)
    
    final_risk_score = min(5, final_risk_score)
    
    st.markdown("## 🧮 Risk Breakdown")
    with st.container():
        st.write("Here is a breakdown of your disaster vulnerability factors:")
        st.write(f"**Model Predicted Score:** {model_score:.2f}")
        st.write(f"**Long-term Risk Score (based on historical data):** {long_term_risk_score:.2f}")
        st.write(f"**Real-Time Risk Score:** {round(real_time_risk_score, 2)}")
    
else:
    st.info("Please enter a country, city, and housing material to calculate risk.")

def disaster_risk_score(disaster_type, severity_level):
    base_scores = {
        "earthquake": 5,
        "flood": 4,
        "cyclone": 6,
        "volcano": 7,
        "wildfire": 6,
        "forest fire": 6
    }
    severity_weight = {
        "green": 1,
        "orange": 2,
        "red": 3
    }
    return base_scores.get(disaster_type.lower(), 2) * severity_weight.get(severity_level.lower(), 1)

    
def get_safety_tips(disaster_type):
    tips = {
        'earthquake': [
            "Drop, Cover, and Hold On.",
            "Stay away from windows.",
            "If outdoors, move to an open area away from buildings."
        ],
        'flood': [
            "Move to higher ground immediately.",
            "Avoid walking or driving through flood waters.",
            "Stay informed through local news."
        ],
        'cyclone': [
            "Secure loose objects around your home.",
            "Stay indoors and away from windows.",
            "Keep emergency supplies ready."
        ],
        'volcano': [
            "Stay indoors and keep windows closed.",
            "Wear masks to avoid inhaling ash.",
            "Prepare for evacuation if advised."
        ],
        'drought': [
            "Conserve water wherever possible.",
            "Avoid unnecessary water usage.",
            "Stay hydrated and monitor health."
        ],
        'wildfire': [
            "Prepare an emergency evacuation plan.",
            "Keep flammable materials away from your home.",
            "Stay updated on fire reports and alerts."
        ]
    }
    if disaster_type is None:
        return ["Stay alert and follow official guidance."]

    return tips.get(disaster_type.lower(), ["Stay alert and follow official guidance."])

safety_tips = get_safety_tips(disaster_type)

if disaster_active:
    st.markdown("### 🛡️ Safety Tips")
    with st.container():
        st.write("Always stay prepared. Here are some safety guidelines:")
        for tip in get_safety_tips(disaster_type):
            st.write(f"- {tip}")
        st.write(f"- Keep your phone charged and emergency contacts saved.")
        st.write(f"- Follow evacuation orders promptly during disasters.")

elif not disaster_active:
    st.markdown("### ✅ No current disasters in your area")
    st.write("Still, it's always a good idea to be prepared! Make sure your emergency kit is ready.")