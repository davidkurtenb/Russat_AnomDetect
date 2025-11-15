import pandas as pd
import numpy as np
import os
import datetime as dt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio

anom_df = pd.read_parquet(r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output\anomDet_CIS_aeanchor_02242021_to_02232024.parquet')

leadup_df = anom_df[(anom_df['datetime']>='08-24-2021') & (anom_df['datetime']<='02-25-2022')]
leadup_df['anom_count'].value_counts()

############ Tech Applications RSOs

tech_app_df = anom_df[anom_df['mission']=='technology_applications']

save_base_dir = r'C:\Users\dk412\Desktop\David\Python Projects\RusSat\output\russat_analysis_artifacts\aeanchor'
save_dir_plots = os.path.join(save_base_dir, 'plots')
save_dir_data = os.path.join(save_base_dir, 'dataframes')

os.makedirs(save_base_dir, exist_ok=True)
os.makedirs(save_dir_plots, exist_ok=True)
os.makedirs(save_dir_data, exist_ok=True)

anom_df['datetime']= pd.to_datetime(anom_df['datetime'], errors='coerce')
anom_df['month']=anom_df['datetime'].dt.month
anom_df['day']=anom_df['datetime'].dt.day
anom_df['day_month']=anom_df['datetime'].dt.strftime('%Y-%m-%d')
anom_df['month_year']=anom_df['datetime'].dt.strftime('%Y-%m')


element = ['inclination',
           'ra_of_asc_node',
           'eccentricity',
           'arg_of_perigee',
           'mean_anomaly',
           'mean_motion']

single_mission_lst =['communications', 
                     'navigation_global_positioning', 
                     'unidentified',
                     'earth_science',
                     'surveillance_and_other_military', 
                     'other',
                     'technology_applications',
                     'solar_physics',
                     'space_physics',        
                     'uncategorized_cosmos', 
                     'engineering',
                     'astronomy', 
                     'planetary_science']

anom_rso_df = anom_df.groupby(['NORAD_CAT_ID','mission'])[['anom_count']].sum()

anom_rso_df=anom_rso_df.reset_index()
anom_rso_df['NORAD_CAT_ID'] = anom_rso_df['NORAD_CAT_ID'].astype('str')
anom_rso_df= anom_rso_df

########### Plot Party

import plotly.express as px
sample_size = 150
fig = px.bar(anom_rso_df.sort_values('anom_count',ascending=False).head(sample_size), 
             x='NORAD_CAT_ID', 
             y='anom_count', 
             color = 'mission')
             #title=f'Total Anomaly Count: Top {sample_size} RSOs',
             #color_discrete_sequence=px.colors.qualitative.Pastel
             
#fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':anom_rso_df['NORAD_CAT_ID']})
#fig.show()
fig.write_html(os.path.join(save_dir_plots,"anomaly_plot.html"))


import plotly.express as px

df = anom_df[anom_df['datetime']<='2022-05-01']
df = df[['month_year', 'anom_count']]
df = df.groupby(['month_year']).agg({'anom_count': 'sum'}).reset_index()
#df.to_excel(os.path.join(save_dir_data,'anomaly_count_by_mission_month.xlsx'))

fig = px.bar(df, x='month_year', y='anom_count', 
              #markers=True,
              title='Anomaly Count by Month',
              color_discrete_sequence=['black'])

fig.add_vline(x="2022-02-24", 
            line_width=2, 
            line_dash="solid", 
            line_color="red")

fig.update_layout(
    legend=dict(
        font=dict(size=16)  
    )
)

fig.update_layout(
    width=1200,   
    height=550,
    xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14))  
)
#fig.show()
fig.write_html(os.path.join(save_dir_plots,"anomaly_count_by_mission_month.png"))

import plotly.express as px
df = anom_df.copy()
#df = anom_df[anom_df['datetime']<='2022-05-01']
df = df[['month_year', 'mission', 'anom_count']]
df = df.groupby(['month_year', 'mission']).agg({'anom_count': 'sum'}).reset_index()
#df.to_excel(os.path.join(save_dir_data,'anomaly_count_by_mission_month.xlsx'))

fig = px.line(df, x='month_year', y='anom_count', 
              color='mission',  
              markers=True,
              #title='Anomaly Count by Mission Over Time',
              color_discrete_sequence=px.colors.qualitative.Bold)

fig.add_vline(x="2022-02-24", 
            line_width=2, 
            line_dash="solid", 
            line_color="red")
fig.update_layout(
    legend=dict(
        font=dict(size=12),
        orientation="h",     
        yanchor="bottom",
        y=-0.4,              
        xanchor="center",
        x=0.5                
    )
)
fig.update_layout(
    width=1200,   
    height=600,
    xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14))  
)
#fig.show()
fig.write_html(os.path.join(save_dir_plots,"anomaly_count_by_mission_month.png"))


post_invasion_df = anom_df[anom_df['datetime'] >= '02-25-2022']
def get_anomaly_rate(df):
    anom_cols = df.columns.str.startswith('anom')
    anom_data = df.iloc[:, anom_cols]
    #df['anom_count'] = anom_data.sum(axis=1)
    df['anomaly_ind'] = (df['anom_count'] >= 1).astype(int)
    #print(f'Total Anomalous Observations for {df} is {df['anomaly_ind'].sum()}')
    total_observations = len(df)
    anomaly_count = len(df[df['anomaly_ind'] == 1])
    return anomaly_count / total_observations if total_observations > 0 else 0

baseline_rate = get_anomaly_rate(post_invasion_df)

result = df.groupby(['month_year', 'mission']).agg({'anom_count': 'sum'}).unstack('mission')
result.columns = result.columns.droplevel(0)  # Removes 'anom_count' level
result.columns.name = None 

cols_to_keep = []
for x in list(result):
    if result[x].sum() > 0:
        cols_to_keep.append(x)

result = result[cols_to_keep]
result.to_excel(os.path.join(save_dir_data,'anomaly_count_by_mission_month.xlsx'))

def plot_all_anom_diff_by_date():
    elements = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    

    fig, axes = plt.subplots(1, len(elements), figsize=(20, 4))


    for i, element in enumerate(elements):
        df = anom_df[
            (anom_df[f'anom_{element}'] > 0) & 
            (anom_df[f'diff_{element}'] != 0) &
            (anom_df['datetime']<'2022-05-01')
        ].copy()
        
        if len(df) > 0:
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_log = np.log10(df_numeric.abs())
            df_log.to_excel(os.path.join(save_dir_data,f'distribuiton_anomalous_changes_{element}.xlsx'))

            
            sns.violinplot(y=df_log[f'diff_{element}'], ax=axes[i])
        
        # Set title and labels
        axes[i].set_title(element.replace('_', ' ').title())
        axes[i].set_ylabel('Log10(|Difference|)' if i == 0 else '')
    
    plt.tight_layout()
    plt.suptitle(f'Distribution of Anomalous Changes', fontsize=14, y=1.02)
    #plt.show()
    plt.savefig(os.path.join(save_dir_data, 'distribution_anomalous_changes.png'),dpi=300, bbox_inches='tight')

plot_all_anom_diff_by_date()


def plot_all_anom_diff_by_mission_separate():
    elements = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    for m in single_mission_lst:
        fig, axes = plt.subplots(1, len(elements), figsize=(20, 4))
        
        mission_df = anom_df[
            (anom_df['mission'] == m) & 
            (anom_df['datetime'] < '2022-05-01')
        ].copy()

        for i, element in enumerate(elements):
            df = mission_df[
                (mission_df[f'anom_{element}'] > 0) & 
                (mission_df[f'diff_{element}'] != 0)
            ].copy()
            
            if len(df) > 0:
                df_numeric = df.apply(pd.to_numeric, errors='coerce')
                df_log = np.log10(df_numeric[f'diff_{element}'].abs())
                df_log.to_excel(os.path.join(save_dir_data,f'distribuiton_anomalous_changes_{m}_{element}.xlsx'))

                
                sns.violinplot(y=df_log, ax=axes[i])
            
            axes[i].set_title(element.replace('_', ' ').title())
            axes[i].set_ylabel('Log10(|Difference|)' if i == 0 else '')
        
        plt.tight_layout()
        plt.suptitle(f'Distribution of Anomalous Changes - {m}', fontsize=14, y=1.02)
        #plt.show()
        plt.savefig(os.path.join(save_dir_data, 'distribution_anomalous_changes_by_mission.png'),dpi=300, bbox_inches='tight')


plot_all_anom_diff_by_mission_separate()


def plot_anomaly_ridgelines():
    elements = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    fig, axes = plt.subplots(len(elements), 1, figsize=(14, 2*len(elements)))
    
    for i, element in enumerate(elements):
        df = anom_df[
            (anom_df[f'anom_{element}'] > 0) & 
            (anom_df[f'diff_{element}'] != 0) &
            (anom_df['datetime']<'2022-05-01')
        ].copy()
        
        df = df.sort_values('month_year')
        
        if len(df) > 0:
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df['log_diff'] = np.log10(df_numeric[f'diff_{element}'].abs())
            
            dates = sorted(df['month_year'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))
            
            for j, date in enumerate(dates):
                date_data = df[df['month_year'] == date]['log_diff']
                if len(date_data) > 1:
                    density = sns.kdeplot(date_data, ax=axes[i], color=colors[j], 
                                        alpha=0.7, label=str(date))
            
            axes[i].set_title(f'{element.replace("_", " ").title()} - Density Over Time')
            axes[i].set_xlabel('Rate of Change (Log10)')
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 7)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_dir_data, 'rate_of_change_ridgeline.png'),dpi=300, bbox_inches='tight')

plot_anomaly_ridgelines()


def plot_anomaly_boxplots():
    elements = ['inclination', 'ra_of_asc_node', 'eccentricity', 'arg_of_perigee', 'mean_anomaly', 'mean_motion']
    
    fig, axes = plt.subplots(3,2, figsize=(25, 20))
    axes = axes.flatten()
    
    for i, element in enumerate(elements):
        df = anom_df[
            (anom_df[f'anom_{element}'] > 0) & 
            (anom_df[f'diff_{element}'] != 0) &
            (anom_df['datetime']<'2022-05-01')
        ].copy()
        
        df = df.sort_values('month_year')
        
        if len(df) > 0:
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df['log_diff'] = np.log10(df_numeric[f'diff_{element}'].abs())
            df.to_excel(os.path.join(save_dir_data,f'distribuiton_anomalous_changes_{element}.xlsx'))

            sns.boxplot(data=df, x='month_year', y='log_diff', ax=axes[i])
            
            y_min, y_max = axes[i].get_ylim()
            y_range = y_max - y_min
            annotation_y = y_max 
            
            counts = df['month_year'].value_counts().sort_index()
            for j, (date, count) in enumerate(counts.items()):
                axes[i].text(j, annotation_y, f'n={count}', 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
    
        axes[i].set_title(element.replace('_', ' ').title(), y = 1.02, fontsize = 12)
        axes[i].set_ylabel('Log10(|Difference|)' if i % 2 == 0 else '')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle('Orbital Element Differincing in Anomaly Distribution', y=1.03, fontsize = 16)
    plt.savefig(os.path.join(save_dir_data, 'boxplot_differencing.png'),dpi=300, bbox_inches='tight')

plot_anomaly_boxplots()