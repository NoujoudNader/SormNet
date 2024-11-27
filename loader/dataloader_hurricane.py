import torch
import numpy as np
import pandas as pd
import os
import math

from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler

from shutil import copyfile



def haversine(lat1, lon1, lat2, lon2):
    '''
    Calculate distance using the Haversine Formula
    '''
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    
    return km

def get_distance(df, station_df): # NN: save time by filling only lower triangle of matrix, upper triangle=lower triangle
    # Ids=df['station_id'].unique() 
    # station_df=create_stationDf(df,Ids, 'offset')
    Ids_new=station_df.columns

    # dist_arr=[]
    dist_arr=np.zeros([len(Ids_new), len(Ids_new)])

    # Loop through each row of the station data DataFrame
    for i in range(len(Ids_new)):
        # dist_i = []
        station_i=df[df['station_id']==Ids_new[i]]
        station_i.reset_index(drop=True,inplace=True)
        lat1=station_i['x'][0]
        lng1=station_i['y'][0]

        # for j in range(len(Ids_new)):
        for j in range(i+1):
            if j > len(Ids_new):
                break
            station_j=df[df['station_id']==Ids_new[j]]
            station_j.reset_index(drop=True,inplace=True)
            #print(station)
            lat2=station_j['x'][0]
            
            lng2=station_j['y'][0]
            
            dist_arr[i,j]=haversine(lat1, lng1, lat2, lng2)
            # dist= haversine(lat1, lng1, lat2, lng2)
            # dist_i.append(dist)
            
        # dist_arr.append(dist_i)

    # Convert dist arrays to numpy arrays
    # dist_arr = np.array(dist_arr)

    # https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix/58806735#58806735
    dist_arr = dist_arr + dist_arr.T - np.diag(np.diag(dist_arr)) 

    return dist_arr


def get_correlation(df, station_df): # NN: save time by filling only lower triangle of matrix, upper triangle=lower triangle
    # Ids=df['station_id'].unique()
    # station_df=create_stationDf(df,Ids, 'offset')
    Ids_new=station_df.columns

    # correlation_mat=[]
    correlation_mat=np.zeros([len(Ids_new), len(Ids_new)])
    # Loop through each row of the station data DataFrame
    for i in range(len(Ids_new)):
        # corr_i = []
        station_i=df[df['station_id']==Ids_new[i]]
        station_i.reset_index(drop=True,inplace=True)

        # for j in range(len(Ids_new)):
        for j in range(i+1):
            if j > len(Ids_new):
                break

            station_j=df[df['station_id']==Ids_new[j]]
            station_j.reset_index(drop=True,inplace=True)
            correlation_mat[i,j] = station_i['observed_data'].corr(station_j['observed_data'])
            # corr = station_i['observed_data'].corr(station_j['observed_data'])
            # corr_i.append(corr)
            
        # correlation_mat.append(corr_i)

    # https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix/58806735#58806735
    correlation_mat = correlation_mat + correlation_mat.T - np.diag(np.diag(correlation_mat)) 


    # Convert dist arrays to numpy arrays
    # correlation_mat = np.array(correlation_mat)
    return correlation_mat

def create_adjancency_matrix(W,Corr, W_mask=1000, Corr_mask=0.75):
    """
    Create an adjacency matrix based on threshold conditions.

    Parameters:
    - W (numpy.ndarray): A distance matrix in km.
    - Corr (numpy.ndarray): A correlation matrix.
    - W_mask (float): The threshold for the distance matrix W.
    - Corr_mask (float): The threshold for the correlations.

    Returns:
    - numpy.ndarray: An adjacency matrix with the correlation values as weights
                     where both weight and correlation conditions are satisfied.
    """
    # Create a mask where both conditions are True
    mask = (W < W_mask) & (np.abs(Corr) > Corr_mask)

    # Create an adjacency matrix with elements from Corr where mask is True, otherwise 0
    adjacency_matrix = np.where(mask, Corr, 0)

    return adjacency_matrix

def create_stationDf(data, Ids, attribute):
    """
    Create a dataframe where each column is a station.

    Parameters:
    - data (dataframe): The original dataframe of hurricane.
    - Ids: stations Ids.
    - attribute: Could be one of these: 'offset', 'observed', 'forecasted' .

    Returns:
    - dataframe: A dataframe containing the 'attribute' data for all stations, 
                    where each station is a column.
    """
    
    # df_station = pd.DataFrame()
    df_station = {}


    for id in Ids:
        sequence=data[data['station_id']==id][attribute] 
        sequence.reset_index(drop=True,inplace=True)
        #print(sequence.shape)
        df_station[id]=sequence

    df_station = pd.DataFrame(df_station)
    df_station=df_station.dropna(axis=1)

    

    return df_station

def sliding_window(station_df, config, sliding_step):
    n_window = config['N_PRED'] + config['N_HIST'] # full window length

    # config['N_Windows']=int(len(station_df)/n_window)  # Works only for non-overlapping windows
    config['N_Windows'] = int((len(station_df) - n_window)/sliding_step)+1

    x = np.ones((station_df.shape[1], config['N_HIST']*config['N_Windows']))*np.nan
    y = np.ones((station_df.shape[1], config['N_PRED']*config['N_Windows']))*np.nan


    # full_window = np.ones((station_df.shape[1], n_window*config['N_Windows']))*np.nan

    # print(n_window)
    # print(sliding_step)
    # print(config['N_Windows'])

    # print(full_window.shape[1])


    for count, i in enumerate(range(0, len(station_df), sliding_step)):

        sta=i
        end = sta + n_window
        
        if end > len(station_df):
            break

        data=np.array(station_df.iloc[sta:end,:].values)
        data=np.swapaxes(data, 0, 1) #data is the offset of all stations
        # print("Data shape: ", data.shape)
        x[:,count*config['N_HIST']:count*config['N_HIST']+config['N_HIST']] = data[:, :config['N_HIST']]
        y[:,count*config['N_PRED']:count*config['N_PRED']+config['N_PRED']] = data[:, config['N_HIST']::]

    # print("y shape: ", y.shape)
    return x, y 


def prepare_gnn_data(df, config, W_mask=1000, Corr_mask=0.7):
    """
    Prepare dataset for GNN 

    Parameters:
    - df (dataframe): The original dataframe of hurricane.
    - config: config object.
    """ 
    Ids=df['station_id'].unique()   
    # print("Ids: ",Ids)
    

    # Split df based on config["SPLITS"]
    df_train, df_val, df_test = get_splits_hurricanes(df, config['SPLITS'])

    # Create station df_* from df_*
    station_df_train =create_stationDf(df_train, Ids, 'offset')
    station_df_val =create_stationDf(df_val, Ids, 'offset')
    station_df_test =create_stationDf(df_test, Ids, 'offset')

    new_common_ids = list(set(station_df_train.columns.values) & set(station_df_val.columns.values) & set(station_df_test.columns.values))
    station_df_train = station_df_train[station_df_train.columns.intersection(new_common_ids)]
    station_df_val = station_df_val[station_df_val.columns.intersection(new_common_ids)]
    station_df_test = station_df_test[station_df_test.columns.intersection(new_common_ids)]

    # id='8726724'
    # y_truth=df_test[df_test['station_id']==id]['offset']
    # print(y_truth)
    # plt.figure()
    # plt.plot(y_truth)
    # plt.show()
    # print("train:", station_df_train)
    # print("test:", station_df_train)
    # print("val:", station_df_train)

    config['N_NODE'] = station_df_train.shape[1]
    # config['N_NODE'] = len(new_common_ids)
    

    # Calculate adjacency matrix based on
    # concatenated station_df_train and station_df_val.
    # Fixed for all timesteps

    # W=get_distance(df, pd.concat([station_df_train, station_df_val], axis=0))
    W=get_distance(df_train, station_df_train)
    # Corr=get_correlation(df, pd.concat([station_df_train, station_df_val], axis=0))
    Corr=get_correlation(df_train, station_df_train)
    adj_matrix=create_adjancency_matrix(W,Corr, W_mask, Corr_mask)
    
    

    _,n_node = W.shape
    n_window = config['N_PRED'] + config['N_HIST'] # full window taken per time t

    # config['N_Windows']=int(len(station_df)/n_window) # Correct only if windows are non-overlapping
    # config['N_SLOT']=24 #hourly data measurment

    # edge_index: 2xnum_edges
    # edge_index = torch.zeros((2, n_node**2), dtype=torch.long)
    # create an edge_attr matrix with our weights  (num_edges x 1) --> our edge features are dim 1
    edge_attr = torch.zeros((n_node**2, 1))
    edge_index = torch.nonzero(torch.from_numpy(adj_matrix)).t()
    mask = edge_index[0] != edge_index[1] #mask to remove self loop edges
    filtered_edge_index = edge_index[:, mask] #remove self loop edges
    edge_index=filtered_edge_index

    edge_attr = adj_matrix[filtered_edge_index[0], filtered_edge_index[1]] #get edges attributes 
    # edge_attr = edge_attr.resize_(edge_attr.shape[0], 1)


    # Apply sliding window
    x_train, y_train = sliding_window(station_df_train, config, 1)
    x_val, y_val = sliding_window(station_df_val, config, 1)
    x_test, y_test = sliding_window(station_df_test, config, config['N_PRED'])
    # check y_test again after windowing
    # print(y_test[0])
    # plt.figure()
    # plt.plot(y_test[0])
    # plt.show()
    # Scale data
    x_scaler = MinMaxScaler().fit(x_train.reshape(-1,1))
    x_train = x_scaler.transform(x_train.reshape(-1,1)).reshape(x_train.shape[0], x_train.shape[1])
    x_val = x_scaler.transform(x_val.reshape(-1,1)).reshape(x_val.shape[0], x_val.shape[1])
    x_test = x_scaler.transform(x_test.reshape(-1,1)).reshape(x_test.shape[0], x_test.shape[1])

    y_scaler = MinMaxScaler().fit(y_train.reshape(-1,1))
    y_train = y_scaler.transform(y_train.reshape(-1,1)).reshape(y_train.shape[0], y_train.shape[1])
    y_val = y_scaler.transform(y_val.reshape(-1,1)).reshape(y_val.shape[0], y_val.shape[1])
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(y_test.shape[0], y_test.shape[1])


    train_gnn = create_temporal_graph(n_node, edge_index, edge_attr, config, len(station_df_train), x_train, y_train)
    val_gnn = create_temporal_graph(n_node, edge_index, edge_attr, config, len(station_df_val), x_val, y_val)
    test_gnn = create_temporal_graph(n_node, edge_index, edge_attr, config, len(station_df_test), x_test, y_test)

    return train_gnn, val_gnn, test_gnn, y_scaler

    # torch.save((data, slices, n_node, mean, std_dev), self.processed_paths[0])


def create_temporal_graph(n_node, edge_index, edge_attr, config, timesteps, x, y):
    
    n_window = config['N_PRED'] + config['N_HIST']
    sequences = []

    # T x F x N
    for i in range(timesteps):
    
        # for each time point construct a different graph with data object
        # Docs here: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        g = Data()
        g.__num_nodes__ = n_node

        g.edge_index = edge_index
        g.edge_attr  = edge_attr

        # Parse windowed data arrays
        # sta=i
        # end = sta + n_window
        
        # if end > timesteps:
        #     break
        # [21, 228]
        # data=np.array(station_df.iloc[sta:end,:].values)
        # full_window = np.swapaxes(data, 0, 1) #data is the offset of station
                    
        # scaler = MinMaxScaler() # Create a scaler object NN: scale before all the data in station_df
        # full_window = scaler.fit_transform(full_window)    # Fit and transform the data
        # full_window = torch.tensor(scaled_data_x_np)     # Convert the scaled data back to tensor

        # g.x = torch.FloatTensor(full_window[:, 0:config['N_HIST']]) #input first n past points
        # g.y = torch.FloatTensor(full_window[:, config['N_HIST']::]) #output predicted after n points

        x_start = i*config['N_HIST']
        x_end = i*config['N_HIST']+config['N_HIST']
        y_start = i*config['N_PRED']
        y_end = i*config['N_PRED']+config['N_PRED']

        if (x_end > x.shape[1]) or (y_end > y.shape[1]):
            break

        g.x = torch.FloatTensor(x[:,x_start:x_end]) #input first n past points
        g.y = torch.FloatTensor(y[:,y_start:y_end]) #output predicted after n points
        sequences += [g]

    # Make the actual dataset
    my_data= sequences
    return my_data

def get_splits_hurricanes(dataset, splits):
    
    """
    Given the data, split it into sequential subsets of train, val, and test as given by splits
    :param dataset: Dataset object to split
    :param splits: [train, val, test] ratios (must be lower than 1 and sum of total =1)
    """
    train = dataset[dataset['storm'].isin(splits['TRAIN_STORMS'])].dropna()
    val = dataset[dataset['storm'].isin(splits['VAL_STORMS'])].dropna()
    test = dataset[dataset['storm'].isin(splits['TEST_STORMS'])].dropna()
    

    return train, val, test
