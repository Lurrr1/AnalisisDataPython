import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
#df_Shunyi = pd.read_csv('../data/PRSA_Data_Shunyi_20130301-20170228.csv')
#df_Tiantan = pd.read_csv('../data/PRSA_Data_Tiantan_20130301-20170228.csv')
#df_Wanliu = pd.read_csv('../data/PRSA_Data_Wanliu_20130301-20170228.csv')

url = 'https://raw.githubusercontent.com/Lurrr1/AnalisisDataPython/refs/heads/main/data/PRSA_Data_Shunyi_20130301-20170228.csv'
df_Shunyi = pd.read_csv(url)
url = 'https://raw.githubusercontent.com/Lurrr1/AnalisisDataPython/refs/heads/main/data/PRSA_Data_Tiantan_20130301-20170228.csv'
df_Tiantan = pd.read_csv(url)
url = 'https://raw.githubusercontent.com/Lurrr1/AnalisisDataPython/refs/heads/main/data/PRSA_Data_Wanliu_20130301-20170228.csv'
df_Wanliu= pd.read_csv(url)

# mengisi missing value dengan nilai rata-rata
df_Shunyi_fill = df_Shunyi.fillna(df_Shunyi.mean(numeric_only=True))
df_Tiantan_fill = df_Tiantan.fillna(df_Tiantan.mean(numeric_only=True))
df_Wanliu_fill = df_Wanliu.fillna(df_Wanliu.mean(numeric_only=True))

# menghilangkan missing value yang tersisa
dfShunyi = df_Shunyi_fill.dropna()
dfTiantan = df_Tiantan_fill.dropna()
dfWanliu = df_Wanliu_fill.dropna()

# membuat fungsi untuk mengubah tipe data
def convert_to_datetime(df):
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    return df

dfShunyi = convert_to_datetime(dfShunyi)
dfTiantan = convert_to_datetime(dfTiantan)
dfWanliu = convert_to_datetime(dfWanliu)

# menghilangkan kolom yang tidak diperlukan
dfShunyi = dfShunyi.drop(columns=['wd','station'])
dfTiantan = dfTiantan.drop(columns=['wd','station'])
dfWanliu = dfWanliu.drop(columns=['wd','station'])

def preprocess_and_aggregate(data, city_name):
    # List of pollutants
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Fill missing values for all pollutants
    for pollutant in pollutants:
        data[pollutant].fillna(data[pollutant].mean(), inplace=True)
    
    # Group by year and calculate mean
    annual_avg = data.groupby('year')[pollutants].mean().reset_index()
    annual_avg['city'] = city_name  # Add city name
    return annual_avg

# Preprocess and aggregate data for each city
shunyi_annual_avg = preprocess_and_aggregate(dfShunyi, 'Shunyi')
tiantan_annual_avg = preprocess_and_aggregate(dfTiantan, 'Tiantan')
wanliu_annual_avg = preprocess_and_aggregate(dfWanliu, 'Wanliu')

# Combine the data for all cities
combined_data = pd.concat([shunyi_annual_avg, tiantan_annual_avg, wanliu_annual_avg], ignore_index=True)

#membuat sidebar untuk memilih jenis analisis
analysis_option = st.sidebar.selectbox(
    "Pilih analisis",("Air Pollution Dashboard (2013-2017)", "Seasonal Trends in PM2.5 Levels Across Stations")
)

if analysis_option == "Air Pollution Dashboard (2013-2017)":
    st.title('Air Pollution Dashboard (2013-2017)')

    # Dropdown for selecting the pollutant
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    pollutant_type = st.selectbox('Select the pollutant to display', pollutants)

    # Function to create bar chart for selected pollutant
    def plot_pollutant_trends(pollutant):
        fig, ax = plt.subplots(figsize=(10, 6))
        for city in combined_data['city'].unique():
            subset = combined_data[combined_data['city'] == city]
            ax.bar(subset['year'] + (0.2 * list(combined_data['city'].unique()).index(city)),
                subset[pollutant], width=0.2, label=city)
        
        # Set chart labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{pollutant} Levels (µg/m³)')
        ax.set_title(f'{pollutant} Levels from 2013 to 2017')
        ax.legend()
        
        # Display the chart in Streamlit
        st.pyplot(fig)

    # Call the function to plot based on the selected pollutant
    plot_pollutant_trends(pollutant_type)

elif analysis_option == "Seasonal Trends in PM2.5 Levels Across Stations":
    st.header("Seasonal Trends in PM2.5 Levels Across Stations")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Mendefinisikan musim
    def assign_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    dfShunyi['season'] = dfShunyi['month'].apply(assign_season)
    dfTiantan['season'] = dfTiantan['month'].apply(assign_season)
    dfWanliu['season'] = dfWanliu['month'].apply(assign_season)

    # Group by season and calculate the mean pollutant values for each station
    seasonal_avg_shunyi = dfShunyi.groupby('season')[pollutants].mean()
    seasonal_avg_tiantan = dfTiantan.groupby('season')[pollutants].mean()
    seasonal_avg_wanliu = dfWanliu.groupby('season')[pollutants].mean()

    # Combine the results for all stations
    seasonal_avg_combined = pd.concat([seasonal_avg_shunyi, seasonal_avg_tiantan, seasonal_avg_wanliu],
                                    keys=['Shunyi', 'Tiantan', 'Wanliu'], names=['Station', 'Season'])

    print(seasonal_avg_combined)

    # Plot seasonal trends for PM2.5 as an example
    seasonal_avg_combined['PM2.5'].unstack(level=0).plot(kind='bar', figsize=(10,6))
    plt.title('Seasonal Trends in PM2.5 Levels Across Stations')
    plt.ylabel('PM2.5 Concentration (µg/m³)')
    plt.xlabel('Season')
    plt.xticks(rotation=0)
    plt.show()
    # Streamlit layout
    st.title("Seasonal Air Quality Analysis")
    st.write("This app displays the seasonal trends of air pollutants across Shunyi, Tiantan, and Wanliu stations.")

    # Plot seasonal trends for PM2.5
    st.subheader("Seasonal Trends in PM2.5 Levels Across Stations")

    # Plotting using Matplotlib
    fig, ax = plt.subplots(figsize=(10,6))
    seasonal_avg_combined['PM2.5'].unstack(level=0).plot(kind='bar', ax=ax)
    ax.set_title('Seasonal Trends in PM2.5 Levels Across Stations')
    ax.set_ylabel('PM2.5 Concentration (µg/m³)')
    ax.set_xlabel('Season')
    ax.set_xticks(range(len(seasonal_avg_combined['PM2.5'].unstack(level=0))))
    ax.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'], rotation=0)
    st.pyplot(fig)

    # Display the combined seasonal average data
    st.subheader("Seasonal Average Pollutant Levels Across Stations")
    st.write(seasonal_avg_combined)
