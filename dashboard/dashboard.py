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

#membuat sidebar untuk memilih jenis analisis
analysis_option = st.sidebar.selectbox(
    "Pilih analisis",("Air Quality Analysis by City", "Seasonal Trends in PM2.5 Levels Across Stations", "RFM Analysis", "RFM Clustering Analysis for Air Quality")
)

if analysis_option == "Air Quality Analysis by City":
# mendefinisikan polusi 
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    # Streamlit UI elements
    st.title("Air Quality Analysis by City (2013-2017)")
    st.subheader("Choose a year and pollutant to display the monthly air quality levels")

    # memasukkan pilihan tahun (2013-2017)
    year = st.selectbox("Select the Year", [2013, 2014, 2015, 2016, 2017])

    # memasukkan pilihan polusi
    pollutant = st.selectbox("Select the Pollutant", pollutants)

    # memasukkan pilihan kota
    city = st.selectbox("Select the City", ['Shunyi', 'Tiantan', 'Wanliu'])

    # membuat fungsi untuk menampilkan filter berdasarkan pilihan user
    def filter_by_year_and_pollutant(df, year, pollutant):
        df_filtered = df[df.index.year == year]
        monthly_pollution = df_filtered.resample('M').mean()[pollutant]
        monthly_pollution.index = monthly_pollution.index.strftime('%B')  # Convert month numbers to month names
        return monthly_pollution

    # Filtering data
    if city == 'Shunyi':
        monthly_pollution = filter_by_year_and_pollutant(dfShunyi, year, pollutant)
    elif city == 'Tiantan':
        monthly_pollution = filter_by_year_and_pollutant(dfTiantan, year, pollutant)
    else:
        monthly_pollution = filter_by_year_and_pollutant(dfWanliu, year, pollutant)

    # Plotting
    st.subheader(f"Monthly {pollutant} Pollution Levels in {city} for {year}")

    # mengindentifikasi peak dan lowest data
    peak_month = monthly_pollution.idxmax()
    lowest_month = monthly_pollution.idxmin()
    peak_value = monthly_pollution.max()
    lowest_value = monthly_pollution.min()

    # menentukan warna chart
    colors = ['blue' if (month != peak_month and month != lowest_month) else 'red' if month == peak_month else 'green' for month in monthly_pollution.index]

    fig, ax = plt.subplots()

    # membuat bar chart berdasarkan data
    monthly_pollution.plot(kind='bar', ax=ax, color=colors)

    # menambahkan label
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{pollutant} Levels")
    ax.set_title(f"Monthly {pollutant} Levels in {city} ({year})")

    # rotasi label pada x-axis
    ax.set_xticklabels(monthly_pollution.index, rotation=45)

    # menambahkan legend
    from matplotlib.lines import Line2D

    # mendefinisikan legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Peak'),
        Line2D([0], [0], color='green', lw=2, label='Lowest'),
        Line2D([0], [0], color='blue', lw=2, label='Average')
    ]

    # menambahkan legend ke plot
    ax.legend(handles=legend_elements, loc='upper right')
    # menampilkan plot
    st.pyplot(fig)

    # menampilkan peak and lowest month and value
    st.write(f"The peak of {pollutant} pollution in {city} was in {year}:")
    st.write(f"Month: {peak_month}, Value: {peak_value:.2f}")
    st.write(f"The Lowest of {pollutant} pollution in {city} was in {year}:")
    st.write(f"Month: {lowest_month}, Value: {lowest_value:.2f}")

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

elif analysis_option == "RFM Analysis":
    st.header("RFM Analysis")
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

    dfShunyi.loc[:, 'datetime'] = pd.to_datetime(dfShunyi[['year', 'month', 'day', 'hour']])
    dfTiantan.loc[:, 'datetime'] = pd.to_datetime(dfTiantan[['year', 'month', 'day', 'hour']])
    dfWanliu.loc[:, 'datetime'] = pd.to_datetime(dfWanliu[['year', 'month', 'day', 'hour']])

    # Define the RFM function
    def calculate_rfm(df, pollution_threshold=100, pollutant_column='PM2.5'):
        # Define pollution event as days where PM2.5 > pollution_threshold
        df['pollution_event'] = df[pollutant_column] > pollution_threshold
        
        # Recency: Days since the last pollution event for each season
        last_event = df.groupby('season')['datetime'].max() - df[df['pollution_event']].groupby('season')['datetime'].max()
        last_event = last_event.dt.days  # Convert to number of days
        
        # Frequency: Number of pollution events per season
        freq = df[df['pollution_event']].groupby('season').size()
        
        # Monetary: Sum of the pollutant values (or you can use mean) - using PM2.5 as an example
        monetary = df.groupby('season')[pollutant_column].sum()
        
        # Combine Recency, Frequency, and Monetary into an RFM matrix
        rfm = pd.DataFrame({
            'Recency': last_event,
            'Frequency': freq,
            'Monetary': monetary
        })
        
        return rfm

        # Example usage for Shunyi, Tiantan, and Wanliu
    rfm_shunyi = calculate_rfm(dfShunyi)
    rfm_tiantan = calculate_rfm(dfTiantan)
    rfm_wanliu = calculate_rfm(dfWanliu)

    # Streamlit interface
    st.title("RFM Analysis for Air Pollution (PM2.5) Events")

    # Let the user select a city
    city = st.selectbox("Select a city", options=['Shunyi', 'Tiantan', 'Wanliu'])

    # Dictionary to map cities to their respective RFM dataframes
    city_rfm_map = {
        'Shunyi': rfm_shunyi,
        'Tiantan': rfm_tiantan,
        'Wanliu': rfm_wanliu
    }

    # Get the RFM data for the selected city
    rfm_selected = city_rfm_map[city]

    # Display the RFM data for the selected city
    st.subheader(f"RFM Analysis for {city}")
    st.write(rfm_selected)

    # Plot bar charts for Recency, Frequency, and Monetary
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Recency Bar Chart
    rfm_selected['Recency'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Recency: Days Since Last Pollution Event')
    axes[0].set_ylabel('Days')
    axes[0].set_xlabel('Season')

    # Frequency Bar Chart
    rfm_selected['Frequency'].plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Frequency: Number of Pollution Events')
    axes[1].set_ylabel('Number of Events')
    axes[1].set_xlabel('Season')

    # Monetary Bar Chart
    rfm_selected['Monetary'].plot(kind='bar', ax=axes[2], color='salmon')
    axes[2].set_title('Monetary: Total PM2.5 Concentration')
    axes[2].set_ylabel('Total PM2.5 (µg/m³)')
    axes[2].set_xlabel('Season')

    # Render the plots in Streamlit
    st.pyplot(fig)
elif analysis_option == "RFM Clustering Analysis for Air Quality":
    st.header("RFM Clustering Analysis for Air Quality")

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

    dfShunyi.loc[:, 'datetime'] = pd.to_datetime(dfShunyi[['year', 'month', 'day', 'hour']])
    dfTiantan.loc[:, 'datetime'] = pd.to_datetime(dfTiantan[['year', 'month', 'day', 'hour']])
    dfWanliu.loc[:, 'datetime'] = pd.to_datetime(dfWanliu[['year', 'month', 'day', 'hour']])

    def calculate_rfm(df, pollution_threshold=100, pollutant_column='PM2.5'):
        # Define pollution event as days where PM2.5 > pollution_threshold
        df['pollution_event'] = df[pollutant_column] > pollution_threshold
        
        # Recency: Days since the last pollution event for each season
        last_event = df.groupby('season')['datetime'].max() - df[df['pollution_event']].groupby('season')['datetime'].max()
        last_event = last_event.dt.days  # Convert to number of days
        
        # Frequency: Number of pollution events per season
        freq = df[df['pollution_event']].groupby('season').size()
        
        # Monetary: Sum of the pollutant values (or you can use mean) - using PM2.5 as an example
        monetary = df.groupby('season')[pollutant_column].sum()
        
        # Combine Recency, Frequency, and Monetary into an RFM matrix
        rfm = pd.DataFrame({
            'Recency': last_event,
            'Frequency': freq,
            'Monetary': monetary
        })
        
        return rfm
        # menentukan bins untuk Recency, Frequency, and Monetary

    rfm_shunyi = calculate_rfm(dfShunyi)
    rfm_tiantan = calculate_rfm(dfTiantan)
    rfm_wanliu = calculate_rfm(dfWanliu)

    recency_bins = [0, 30, 90, 180]  # Days since last pollution event
    frequency_bins = [0, 5, 15, 30]  # Number of pollution events per season
    monetary_bins = [0, 500, 1000, 2000]  # PM2.5 sum

    # menggunakan bins
    rfm_shunyi['RecencyCluster'] = pd.cut(rfm_shunyi['Recency'], bins=recency_bins, labels=['Very Recent', 'Recent', 'Distant'])
    rfm_shunyi['FrequencyCluster'] = pd.cut(rfm_shunyi['Frequency'], bins=frequency_bins, labels=['Low', 'Medium', 'High'])
    rfm_shunyi['MonetaryCluster'] = pd.cut(rfm_shunyi['Monetary'], bins=monetary_bins, labels=['Low', 'Medium', 'High'])

    rfm_tiantan['RecencyCluster'] = pd.cut(rfm_tiantan['Recency'], bins=recency_bins, labels=['Very Recent', 'Recent', 'Distant'])
    rfm_tiantan['FrequencyCluster'] = pd.cut(rfm_tiantan['Frequency'], bins=frequency_bins, labels=['Low', 'Medium', 'High'])
    rfm_tiantan['MonetaryCluster'] = pd.cut(rfm_tiantan['Monetary'], bins=monetary_bins, labels=['Low', 'Medium', 'High'])

    rfm_wanliu['RecencyCluster'] = pd.cut(rfm_wanliu['Recency'], bins=recency_bins, labels=['Very Recent', 'Recent', 'Distant'])
    rfm_wanliu['FrequencyCluster'] = pd.cut(rfm_wanliu['Frequency'], bins=frequency_bins, labels=['Low', 'Medium', 'High'])
    rfm_wanliu['MonetaryCluster'] = pd.cut(rfm_wanliu['Monetary'], bins=monetary_bins, labels=['Low', 'Medium', 'High'])

    def process_rfm_clusters(rfm):
        # Add 'Unknown' as a category to the clusters
        rfm['RecencyCluster'] = rfm['RecencyCluster'].cat.add_categories('Unknown')
        rfm['FrequencyCluster'] = rfm['FrequencyCluster'].cat.add_categories('Unknown')
        rfm['MonetaryCluster'] = rfm['MonetaryCluster'].cat.add_categories('Unknown')

        # Fill missing values with 'Unknown'
        rfm['RecencyCluster'].fillna('Unknown', inplace=True)
        rfm['FrequencyCluster'].fillna('Unknown', inplace=True)
        rfm['MonetaryCluster'].fillna('Unknown', inplace=True)

        return rfm

    # Function to plot the RFM scatter plot
    def plot_rfm_scatter(rfm, city_name):
        colors = {'Very Recent': 'red', 'Recent': 'blue', 'Distant': 'green', 'Unknown': 'gray'}
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with Recency vs Frequency using color mapping based on RecencyCluster
        plt.scatter(rfm['Recency'], rfm['Frequency'],
                    c=rfm['RecencyCluster'].map(colors),
                    s=rfm['Monetary'] / 10, alpha=0.6)

        plt.title(f'RFM Clustering (Recency vs Frequency) for {city_name}')
        plt.xlabel('Recency (Days Since Last Event)')
        plt.ylabel('Frequency (Number of Events)')
        st.pyplot(plt)

    # Streamlit app layout
    st.title("RFM Clustering Analysis for Air Quality")
    city = st.selectbox("Select a city", options=['Shunyi', 'Tiantan', 'Wanliu'])

    # Assuming RFM data has already been calculated and clusters assigned
    rfm_shunyi = process_rfm_clusters(rfm_shunyi)
    rfm_tiantan = process_rfm_clusters(rfm_tiantan)
    rfm_wanliu = process_rfm_clusters(rfm_wanliu)

    # City mapping for RFM data
    city_rfm_map = {
        'Shunyi': rfm_shunyi,
        'Tiantan': rfm_tiantan,
        'Wanliu': rfm_wanliu
    }

    # Get the selected city's RFM data
    rfm_selected = city_rfm_map[city]

    # Display the scatter plot based on the selected city
    st.subheader(f"RFM Clustering Analysis for {city}")
    plot_rfm_scatter(rfm_selected, city)