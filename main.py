from datetime import datetime
start=datetime.now()

from loader.dataloader_hurricane import *

data_df=pd.read_csv("STOFSatl_hydro.csv", low_memory=False).dropna()

# Remove outliers
std = data_df['offset'].std()
data_df = data_df.where(data_df['offset'].abs() < 3*np.abs(std)).dropna()


storm_list = ['CHARLEY', 'HERMINE', 'IDALIA', 'IAN', 'DENNIS', 'WILMA', 'DEBBY', 'MICHAEL', 'ETA', 'FRED', 'HELENE', 'MILTON']
data_df = data_df[data_df['storm'].isin(storm_list)].dropna()

# Keep only stations in the Gulf 
data_df = data_df[(data_df['x'] < -81.875)].dropna()



# Find common station IDs through all storms

storms = data_df['storm'].unique()
ids_per_storm = {}
for storm in storms:
    ids_per_storm[storm] = data_df[data_df['storm']==storm]['station_id'].unique().tolist()

common_ids = list(set.intersection(*map(set, [ids_per_storm[storm] for storm in storms])))
print("No. of common stations before droping NaN values:")
print(len(common_ids))

# station_info_df = data_df[data_df['station_id'].isin(common_ids)][['x', 'y', 'station_id', 'agency']].drop_duplicates()
# print('Agencies of unique stations:')
# print(station_info_df['agency'].unique())


import matplotlib.pyplot as plt
import matplotlib as mpl

import cartopy.crs as crs
import cartopy.feature as cfeature

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(crs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = crs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
    
    # Constant config to use throughout
config = {
    'BATCH_SIZE': 20,
    'EPOCHS': 200,
    'WEIGHT_DECAY': 5e-7, 
    'INITIAL_LR': 3e-5,   
    'CHECKPOINT_DIR': './runs',
    'N_PRED': 48,
    'N_HIST': 45,
    'DROPOUT': 0.4,     
    # number of possible 5 minute measurements per day
    # number of days worth of data in the dataset
    # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
    'USE_GAT_WEIGHTS': True,
    'SPLITS': {
    'TRAIN_STORMS': ['CHARLEY', 'HERMINE', 'DENNIS', 'WILMA', 'DEBBY', 'MICHAEL', 'ETA', 'FRED', 'HELENE', 'MILTON'],
    'VAL_STORMS': ['IAN'],
    'TEST_STORMS': ['IDALIA'],
    }
}

station_df=create_stationDf(data_df, common_ids, 'offset')
config['N_NODE'] = station_df.shape[1]
print("Number of common stations after droping NaN values\n", len(station_df.columns.values))

df_train, df_val, df_test = get_splits_hurricanes(data_df, config['SPLITS'])

# Create station df_* from df_*
station_df_train =create_stationDf(df_train, common_ids, 'offset')
station_df_val =create_stationDf(df_val, common_ids, 'offset')
station_df_test =create_stationDf(df_test, common_ids, 'offset')

new_common_ids = set.intersection(set(station_df_train.columns.values), set(station_df_val.columns.values), set(station_df_test.columns.values))
station_df_train = station_df_train[station_df_train.columns.intersection(new_common_ids)]
station_df_val = station_df_val[station_df_val.columns.intersection(new_common_ids)]
station_df_test = station_df_test[station_df_test.columns.intersection(new_common_ids)]


station_info_df = data_df[data_df['station_id'].isin(station_df.columns.values)][['x', 'y', 'station_id', 'agency']].drop_duplicates()
print('Agencies of unique stations:')
print(station_info_df['agency'].unique())


station_info_df_NOOA = station_info_df[(station_info_df['agency'] == 'NOAA_NOS') | (station_info_df['agency'] == 'TCOON')]
station_info_df_USGS = station_info_df[(station_info_df['agency'] == 'USGS') | (station_info_df['agency'] == 'USACE')]



plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 20})

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.add_feature(cfeature.LAND, color="lightgrey")
ax.add_feature(cfeature.LAKES, color="dodgerblue")
ax.add_feature(cfeature.BORDERS, linestyle="--")
ax.add_feature(cfeature.OCEAN, color="dodgerblue")
ax.add_feature(cfeature.RIVERS, color="dodgerblue")
ax.add_feature(cfeature.STATES)
ax.gridlines()

im1 = ax.scatter(station_info_df_NOOA['x'], station_info_df_NOOA['y'], c = 'green',
                  edgecolors='black',
                    s=75,
                      alpha=1.0,
                        label='NOAA-NOS')
im2 = ax.scatter(station_info_df_USGS['x'], station_info_df_USGS['y'], c = 'red',
                  marker = 's',
                    edgecolors='black',
                      s=75, alpha=1.0,
                        label='USGS-USACE')

plt.xlim((station_info_df['x'].min()-3, station_info_df['x'].max()+3))
plt.ylim((station_info_df['y'].min()-3, station_info_df['y'].max()+3))

plt.legend()
scale_bar(ax, 500)
plt.title('Common stations in STOFSatl runs')
# plt.show()
plt.savefig('common_stations.pdf', bbox_inches='tight', dpi=300)
plt.close()

train_gnn, val_gnn, test_gnn, scaler=prepare_gnn_data(data_df, config, 500, 0.7)

import networkx as nx
from torch_geometric.utils import to_networkx

g = to_networkx(train_gnn[0], to_undirected=True)
# nx.draw(g, with_labels=True)
# plt.show()

# g = to_networkx(test_gnn[0])
nx.draw_circular(g, with_labels=True)
plt.savefig('circular_graph.pdf', bbox_inches='tight',dpi=300)
plt.close()

# mapping = {}
positions = {}
names = {}
ids = {}
agencies = {}

for node_name in g.nodes:
    x = data_df['x'].where(data_df['station_id']==str(station_df_train.columns[node_name])).dropna().unique()[0]
    y = data_df['y'].where(data_df['station_id']==str(station_df_train.columns[node_name])).dropna().unique()[0]
    names[node_name] = data_df['station_name'].where(data_df['station_id']==str(station_df_train.columns[node_name])).dropna().unique()[0]
    ids[node_name] = data_df['station_id'].where(data_df['station_id']==str(station_df_train.columns[node_name])).dropna().unique()[0]
    agencies[node_name] = data_df['agency'].where(data_df['station_id']==str(station_df_train.columns[node_name])).dropna().unique()[0]
    positions[node_name] = (x,y)

plt.rcParams['figure.figsize'] = [25, 10]
plt.rcParams.update({'font.size': 20})

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.COASTLINE, edgecolor="black")
ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.add_feature(cfeature.LAND, color="lightgrey")
ax.add_feature(cfeature.LAKES, color="dodgerblue")
ax.add_feature(cfeature.BORDERS, linestyle="--")
ax.add_feature(cfeature.OCEAN, color="dodgerblue")
ax.add_feature(cfeature.RIVERS, color="dodgerblue")
ax.add_feature(cfeature.STATES)
ax.gridlines()

d = dict(g.degree)
low, *_, high = sorted(d.values())
norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

nx.draw(g, pos=positions, with_labels=True, node_color=[mapper.to_rgba(i) for i in d.values()])

plt.xlim((station_info_df['x'].min()-1.5, station_info_df['x'].max()+1.5))
plt.ylim((station_info_df['y'].min()-1.5, station_info_df['y'].max()+1.5))

scale_bar(ax, 500)
fig.colorbar(mapper, ax=ax)
plt.savefig('geomap_graph.pdf', bbox_inches='tight',dpi=300)
plt.close()

from torch_geometric.loader import DataLoader

train_dataloader = DataLoader(train_gnn, batch_size=config['BATCH_SIZE'], shuffle=True)
val_dataloader = DataLoader(val_gnn, batch_size=config['BATCH_SIZE'], shuffle=True)
test_dataloader = DataLoader(test_gnn, batch_size=14, shuffle=False)

from models.trainer import load_from_checkpoint, model_train, model_test
from torch_geometric.loader import DataLoader


 # Get gpu if you can
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# Configure and train model
config['N_NODE'] = train_gnn[0].x.shape[0]
print(config['N_NODE'])
model = model_train(train_dataloader, val_dataloader, config, device)

predictions = []
ground_truths = []

for i, batch in enumerate(test_dataloader):
    batch = batch.to(device)
    print(batch.size)
    if batch.x.shape[0] == 1:
        pass
    else:
        with torch.no_grad():
            pred = model[0](batch, device)

        truth = batch.y.to(device)  # Ensure it's on the same device

        predictions.append(pred)
        ground_truths.append(truth)

        


predictions = torch.cat(predictions, dim=0).cpu()
ground_truths = torch.cat(ground_truths, dim=0).cpu()
num_nodes = config['N_NODE']  # Number of nodes per graph = 74-1
num_features = predictions.shape[1]  # Number of features per node
num_graphs = predictions.shape[0] // num_nodes  # Total number of graphs

# Reshape predictions
reshaped_predictions = predictions.view(num_graphs, num_nodes, num_features)
reshaped_truth=ground_truths.view(num_graphs, num_nodes, num_features)

reshaped_predictions = scaler.inverse_transform(reshaped_predictions.reshape(-1,1)).reshape(num_graphs, num_nodes, num_features)
reshaped_truth = scaler.inverse_transform(reshaped_truth.reshape(-1,1)).reshape(num_graphs, num_nodes, num_features)

# Print global stats

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_squared_error

plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 300
plt.scatter(reshaped_truth.flatten(), reshaped_predictions.flatten(), edgecolors='black', s=20)
plt.axline((0,0), slope=1, c='red', linestyle='dashed')
plt.xlabel('Real offsets (ft.)')
plt.ylabel('Predicted offsets (ft.)')
plt.xlim(-2, 1)
plt.ylim(-2, 1)
r_sq_str = "R\u00b2 = {:.3f}\n".format(r2_score(reshaped_truth.flatten(), reshaped_predictions.flatten()))
mse_str = "MSE = {:.3f}\n".format(mean_squared_error(reshaped_truth.flatten(), reshaped_predictions.flatten()))
rmse_str = "RMSE = {:.3f}\n".format(root_mean_squared_error(reshaped_truth.flatten(), reshaped_predictions.flatten()))
mae_str = "MAE = {:.3f}\n".format(mean_absolute_error(reshaped_truth.flatten(), reshaped_predictions.flatten()))
stat_str = r_sq_str + mse_str + rmse_str + mae_str
plt.annotate(stat_str, xy=(0.70, 0.05), xycoords='axes fraction')
plt.savefig('real_vs_pred_offsets_all.pdf', bbox_inches='tight',dpi=300)
plt.close()
print(stat_str)

import os

plt.rcParams['figure.figsize'] = [6.4, 3.6]
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 300

if not os.path.exists('real_vs_obs_offsets'):
   os.makedirs('real_vs_obs_offsets')

if not os.path.exists('water_levels'):
   os.makedirs('water_levels')
   
if not os.path.exists('csv'):
   os.makedirs('csv')

# Initialize lists for global water level stats

observed_data_global = []
forecast_data_global = []
corr_forecast_data_global = []

for node_index in g:
   # Extract the predictions for the specific node across all graphs
   node_pred = reshaped_predictions[:, node_index, :].reshape(-1)  # Shape: [num_graphs, num_features]
   node_truth = reshaped_truth[:, node_index, :].reshape(-1)  # Shape: [num_graphs, num_features]


   # Plot offsets
   plt.figure()
   plt.plot(node_pred,label='GNN')
   plt.plot(node_truth, label='Observed')
   plt.legend()
   title_str = f'{names[node_index]} ({ids[node_index]}, {agencies[node_index].replace("_", "-")})'
   plt.title(title_str)

   r_sq_str = "R\u00b2 = {:.3f}\n".format(r2_score(node_truth, node_pred))
   mse_str = "MSE = {:.3f}\n".format(mean_squared_error(node_truth, node_pred))
   rmse_str = "RMSE = {:.3f}\n".format(root_mean_squared_error(node_truth, node_pred))
   mae_str = "MAE = {:.3f}\n".format(mean_absolute_error(node_truth, node_pred))
   stat_str = r_sq_str + mse_str + rmse_str + mae_str
   plt.annotate(stat_str, xy=(0.75, 0.05), xycoords='axes fraction')


   plt.ylabel('Offset (ft.)')
   plt.xlabel('Time (min.)')
   save_str = os.path.join('real_vs_obs_offsets', str(ids[node_index])) + '.pdf'
   plt.savefig(save_str, dpi=300)
   plt.close()

   # Plot water levels
   test_storm = config['SPLITS']['TEST_STORMS'][0]
   id = ids[node_index]

   station_df_test = create_stationDf(data_df[(data_df['storm']==test_storm)], ids, 'offset')

   observed_data = data_df[(data_df['storm']==test_storm) & (data_df['station_id']==id) & (data_df['time_UTC'].isin(station_df_test.index.values))]['observed_data'].values
   forecast_data = data_df[(data_df['storm']==test_storm) & (data_df['station_id']==id) & (data_df['time_UTC'].isin(station_df_test.index.values))]['forecast_data'].values

   node_pred = np.concat((np.ones(config['N_HIST'])*np.nan, node_pred))


   if len(forecast_data) > len(node_pred):
      node_pred = np.concat((node_pred, np.ones(len(forecast_data)-len(node_pred))*np.nan))

   corr_forecast_data = forecast_data - node_pred

   df_eval = pd.DataFrame({'observed_data': observed_data, 'forecast_data': forecast_data, 'predicted_offset':node_pred, 'corr_forecast_data': corr_forecast_data}).dropna()
   save_str = os.path.join('csv', str(ids[node_index])) + '.csv'
   df_eval.to_csv(save_str)

   observed_data_global.append(observed_data.tolist())
   forecast_data_global.append(forecast_data.tolist())
   corr_forecast_data_global.append(corr_forecast_data.tolist())

   plt.figure()
   plt.plot(observed_data,label='Observed')
   plt.plot(forecast_data, label='Forecast')
   plt.plot(corr_forecast_data, label='Forecast+ML')
   plt.xlabel('Time (min.)')
   plt.ylabel('Water level (ft.)')
   plt.legend(loc='upper left', fontsize='small')
   title_str = f'{names[node_index]} ({ids[node_index]}, {agencies[node_index].replace("_", "-")})'
   plt.title(title_str)

   r_sq_obs_for = r2_score(observed_data, forecast_data)
   mse_obs_for = mean_squared_error(observed_data, forecast_data)
   rmse_obs_for = root_mean_squared_error(observed_data, forecast_data)
   mae_obs_for = mean_absolute_error(observed_data, forecast_data)


   # stats for observed - forecast
   r_sq_obs_for = r2_score(df_eval['observed_data'], df_eval['forecast_data'])
   mse_obs_for = mean_squared_error(df_eval['observed_data'], df_eval['forecast_data'])
   rmse_obs_for = root_mean_squared_error(df_eval['observed_data'], df_eval['forecast_data'])
   mae_obs_for = mean_absolute_error(df_eval['observed_data'], df_eval['forecast_data'])


   # stats for observed - corr. forecast
   r_sq_obs_corr = r2_score(df_eval['observed_data'], df_eval['corr_forecast_data'])
   mse_obs_corr = mean_squared_error(df_eval['observed_data'], df_eval['corr_forecast_data'])
   rmse_obs_corr = root_mean_squared_error(df_eval['observed_data'], df_eval['corr_forecast_data'])
   mae_obs_corr = mean_absolute_error(df_eval['observed_data'], df_eval['corr_forecast_data'])

   evaluation_stats = [r_sq_obs_for, mse_obs_for, rmse_obs_for, mae_obs_for, r_sq_obs_corr, mse_obs_corr, rmse_obs_corr, mae_obs_corr]


   # stats for observed - corr. forecast
   # stats for observed - forecast
   r_sq_str = "R\u00b2 = {:.3f}\n".format(evaluation_stats[0])
   mse_str = "MSE = {:.3f}\n".format(evaluation_stats[1])
   rmse_str = "RMSE = {:.3f}\n".format(evaluation_stats[2])
   mae_str = "MAE = {:.3f}\n".format(evaluation_stats[3])
   stat_str = 'Without ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
   annotation = plt.annotate(stat_str, xy=(0.05, 0.025), xycoords='axes fraction', fontsize='small')
   annotation.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))

   # stats for observed - corr. forecast
   r_sq_str = "R\u00b2 = {:.3f}\n".format(evaluation_stats[4])
   mse_str = "MSE = {:.3f}\n".format(evaluation_stats[5])
   rmse_str = "RMSE = {:.3f}\n".format(evaluation_stats[6])
   mae_str = "MAE = {:.3f}\n".format(evaluation_stats[7])
   stat_str = 'With ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
   annotation = plt.annotate(stat_str, xy=(0.75, 0.025), xycoords='axes fraction', fontsize='small')
   annotation.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
   save_str = os.path.join('water_levels', str(ids[node_index])) + '.pdf'
   plt.savefig(save_str, dpi=300)
   plt.close()

# Global water level stats 
observed_data_global = np.array(observed_data_global).flatten()
forecast_data_global = np.array(forecast_data_global).flatten()
corr_forecast_data_global = np.array(corr_forecast_data_global).flatten()
df_eval = pd.DataFrame({'observed_data': observed_data_global, 'forecast_data': forecast_data_global, 'corr_forecast_data': corr_forecast_data_global}).dropna()

# stats for observed - forecast
r_sq_obs_for = r2_score(df_eval['observed_data'], df_eval['forecast_data'])
mse_obs_for = mean_squared_error(df_eval['observed_data'], df_eval['forecast_data'])
rmse_obs_for = root_mean_squared_error(df_eval['observed_data'], df_eval['forecast_data'])
mae_obs_for = mean_absolute_error(df_eval['observed_data'], df_eval['forecast_data'])


# stats for observed - corr. forecast
r_sq_obs_corr = r2_score(df_eval['observed_data'], df_eval['corr_forecast_data'])
mse_obs_corr = mean_squared_error(df_eval['observed_data'], df_eval['corr_forecast_data'])
rmse_obs_corr = root_mean_squared_error(df_eval['observed_data'], df_eval['corr_forecast_data'])
mae_obs_corr = mean_absolute_error(df_eval['observed_data'], df_eval['corr_forecast_data'])

evaluation_stats = [r_sq_obs_for, mse_obs_for, rmse_obs_for, mae_obs_for, r_sq_obs_corr, mse_obs_corr, rmse_obs_corr, mae_obs_corr]

# Print
# stats for observed - forecast
r_sq_str = "R\u00b2 (global) = {:.3f}\n".format(evaluation_stats[0])
mse_str = "MSE (global) = {:.3f}\n".format(evaluation_stats[1])
rmse_str = "RMSE (global) = {:.3f}\n".format(evaluation_stats[2])
mae_str = "MAE (global) = {:.3f}\n".format(evaluation_stats[3])
stat_str = 'Without ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
print(stat_str)

# stats for observed - corr. forecast
r_sq_str = "R\u00b2 (global) = {:.3f}\n".format(evaluation_stats[4])
mse_str = "MSE (global) = {:.3f}\n".format(evaluation_stats[5])
rmse_str = "RMSE (global) = {:.3f}\n".format(evaluation_stats[6])
mae_str = "MAE (global) = {:.3f}\n".format(evaluation_stats[7])
stat_str = 'With ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
print(stat_str)

plt.figure()
plt.plot(observed_data_global,label='Observed')
plt.plot(forecast_data_global, label='Forecast')
plt.plot(corr_forecast_data_global, label='Forecast+ML')
plt.xlabel('Time (min.)')
plt.ylabel('Water level (ft.)')
plt.legend(loc='upper left', fontsize='small')
# stats for observed - corr. forecast
# stats for observed - forecast
r_sq_str = "R\u00b2 = {:.3f}\n".format(evaluation_stats[0])
mse_str = "MSE = {:.3f}\n".format(evaluation_stats[1])
rmse_str = "RMSE = {:.3f}\n".format(evaluation_stats[2])
mae_str = "MAE = {:.3f}\n".format(evaluation_stats[3])
stat_str = 'Without ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
annotation = plt.annotate(stat_str, xy=(0.05, 0.025), xycoords='axes fraction', fontsize='small')
annotation.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))

# stats for observed - corr. forecast
r_sq_str = "R\u00b2 = {:.3f}\n".format(evaluation_stats[4])
mse_str = "MSE = {:.3f}\n".format(evaluation_stats[5])
rmse_str = "RMSE = {:.3f}\n".format(evaluation_stats[6])
mae_str = "MAE = {:.3f}\n".format(evaluation_stats[7])
stat_str = 'With ML:\n' + r_sq_str + mse_str + rmse_str + mae_str
annotation = plt.annotate(stat_str, xy=(0.75, 0.025), xycoords='axes fraction', fontsize='small')
annotation.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
plt.savefig('Water_levels_global.pdf')
plt.close()

print('Time', datetime.now()-start)
