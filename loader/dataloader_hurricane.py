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

def get_distance(df): # NN: save time by filling only lower triangle of matrix, upper triangle=lower triangle
    Ids=df['station_id'].unique() 
    station_df=create_stationDf(df,Ids, 'offset')
    Ids_new=station_df.columns

    dist_arr=[]
    # Loop through each row of the station data DataFrame
    for i in range(len(Ids_new)):
        dist_i = []
        station_i=df[df['station_id']==Ids_new[i]]
        station_i.reset_index(drop=True,inplace=True)
        lat1=station_i['x'][0]
        lng1=station_i['y'][0]

        for j in range(len(Ids_new)):
            station=df[df['station_id']==Ids_new[j]]
            station.reset_index(drop=True,inplace=True)
            #print(station)
            lat2=station['x'][0]
            
            lng2=station['y'][0]
            
            dist= haversine(lat1, lng1, lat2, lng2)
            dist_i.append(dist)
            
        dist_arr.append(dist_i)

    # Convert dist arrays to numpy arrays
    dist_arr = np.array(dist_arr)
    return dist_arr


def get_correlation(df): # NN: save time by filling only lower triangle of matrix, upper triangle=lower triangle
    Ids=df['station_id'].unique()
    station_df=create_stationDf(df,Ids, 'offset')
    Ids_new=station_df.columns

    correlation_mat=[]
    # Loop through each row of the station data DataFrame
    for i in range(len(Ids_new)):
        corr_i = []
        station_i=df[df['station_id']==Ids_new[i]]
        station_i.reset_index(drop=True,inplace=True)

        for j in range(len(Ids_new)):
            station_j=df[df['station_id']==Ids_new[j]]
            station_j.reset_index(drop=True,inplace=True)
            corr = station_i['observed_data'].corr(station_j['observed_data'])
            corr_i.append(corr)
            
        correlation_mat.append(corr_i)

    # Convert dist arrays to numpy arrays
    correlation_mat = np.array(correlation_mat)
    return correlation_mat

def create_adjancency_matrix(W,Corr, W_mask=1000, Corr_mask=0.7):
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
    mask = (W < W_mask) & (Corr > Corr_mask)

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
    df_station = pd.DataFrame()
    for id in Ids:
        sequence=data[data['station_id']==id][attribute] 
        sequence.reset_index(drop=True,inplace=True)
        #print(sequence.shape)
        df_station[id]=sequence
    
    df_station=df_station.dropna(axis=1)
    return df_station

def prepare_gnn_data(df, config, W_mask=1000, Corr_mask=0.7):
        """
        Prepare dataset for GNN 

        Parameters:
        - df (dataframe): The original dataframe of hurricane.
        - config: config object.
        """ 

        W=get_distance(df)
        Corr=get_correlation(df)
        adj_matrix=create_adjancency_matrix(W,Corr)
        Ids=df['station_id'].unique()   
        station_df=create_stationDf(df,Ids, 'offset')
        

        _,n_node = W.shape
        n_window = config['N_PRED'] + config['N_HIST'] # full window taken per time t

        config['N_Windows']=int(len(station_df)/n_window)
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

        sequences = []
 

        # T x F x N
        for i in range(len(station_df)):
        
            # for each time point construct a different graph with data object
            # Docs here: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
            g = Data()
            g.__num_nodes__ = n_node

            g.edge_index = edge_index
            g.edge_attr  = edge_attr

            # (F,N) switched to (N,F)
            sta=i
            end = sta + n_window
            
            if end > len(station_df)-1:
                break
            # [21, 228]
            data=np.array(station_df.iloc[sta:end,:].values)
            full_window = np.swapaxes(data, 0, 1) #data is the offset of station
                        
            scaler = MinMaxScaler() # Create a scaler object NN: scale before all the data in station_df
            full_window = scaler.fit_transform(full_window)    # Fit and transform the data
            # full_window = torch.tensor(scaled_data_x_np)     # Convert the scaled data back to tensor

            g.x = torch.FloatTensor(full_window[:, 0:config['N_HIST']]) #input first n past points
            g.y = torch.FloatTensor(full_window[:, config['N_HIST']::]) #output predicted after n points
            sequences += [g]

        # Make the actual dataset
        my_data= sequences
        return my_data
        # torch.save((data, slices, n_node, mean, std_dev), self.processed_paths[0])


def get_splits_hurricanes(dataset, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: Dataset object to split
    :param splits: [train, val, test] ratios (must be lower than 1 and sum of total =1)
    """
    split_train, split_val, _ = splits
    n_total=len(dataset)
    i = int(n_total*split_train)
    j = int(n_total*split_val)
    train = dataset[:i]
    val = dataset[i:i+j]
    test = dataset[i+j:]

    return train, val, test