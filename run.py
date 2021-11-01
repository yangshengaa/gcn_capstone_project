"""
run the two baseline models and report respective training accuracies 
"""

# import packages 
import configparser

# load files 
from src.data_reader import read_data
from src.data_downloader import download_all_data
from src.data_preprocessor import *
from src.baseline_models import *

# =====================
# ---- load config ----
# =====================

def load_config():
    """ return config and print """
    # load config 
    config = configparser.ConfigParser()
    config.read('config.ini')
    # print 
    print(f"Selected Dataset: {config['DATASET']['source']}")
    print(f"Is Undirected:    {config['DATASET']['convert_to_undirected']}")
    print(f"Model Type:       {config['DATASET']['model_type']}")
    print()
    print(f"Hidden Layer Dimension: {config['DEFAULT']['hidden_layer_dim']}")
    print(f"Max Epochs:             {config['DEFAULT']['max_epochs']}")
    print(f"Learning Rate:          {config['DEFAULT']['learning_rate']}")
    print()
    return config

# =====================
# ------- run ---------
# =====================

if __name__ == '__main__':
    # specify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using Device: {device}')
    
    print('------- Download Data --------')
    download_all_data()

    # load config
    print('------- Loading Config -------')
    cfg = load_config()

    # load data 
    print('--------- Read Data ----------')
    features, labels, adjacency_matrix = read_data(cfg['DATASET']['source'])
    # set up batch features and labels, and specify loss function according to model type
    batch_features = torch.Tensor([features.numpy()]).to(device)
    model_type = cfg['DATASET']['model_type']
    input_shape = features.shape[1] 
    if model_type == 'classification':
        output_shape = len(np.unique(labels))  # number of classes 
        loss_fn = nn.CrossEntropyLoss()        # Cross Entropy for Classification
        batch_labels = labels.to(device)
    elif model_type == 'regression':
        output_shape = 1                       # one output 
        loss_fn = nn.MSELoss()                 # MSE for Regression
        batch_labels = torch.Tensor([labels.numpy()]).T.to(device)
    else:
        raise NotImplementedError('Model Type must be either classification or regression')

    # preprocess, or build graph 
    print('--------- Build Graph --------')
    if eval(cfg['GRAPH_BUILD']['build_graph']): 
        builder = cfg['GRAPH_BUILD']['graph_builder']  # select the graph builder
        adjacency_matrix = eval(f"{builder}(features)")
    else:
        print('Graph Already Loaded')

    # symmetrize 
    if eval(cfg['DATASET']['convert_to_undirected']):
        adjacency_matrix = symmetrize_adjacency_matrix(adjacency_matrix)
    
    # report training accuracy 
    print('-------- Model Training ------')
    # parameters 
    max_epochs = int(cfg['DEFAULT']['max_epochs'])
    learning_rate = float(cfg['DEFAULT']['learning_rate'])
    hidden_layer_dim = int(cfg['DEFAULT']['hidden_layer_dim'])
    is_classification = model_type == 'classification'

    # fcn 
    print('FCN: ')
    fcn = FCN(
        input_shape=input_shape, 
        output_shape=output_shape, 
        hidden_layer_dim=hidden_layer_dim
    ).to(device)
    train_loop(
        fcn, batch_features, batch_labels, loss_fn, 
        max_epoch=max_epochs, learning_rate=learning_rate, is_classification=is_classification
    )
    print()

    # gcn 
    print('GCN: ')
    gcn = GCN(
        input_shape=input_shape,
        output_shape=output_shape,
        adjacency_matrix=adjacency_matrix,
        hidden_layer_dim=hidden_layer_dim
    ).to(device)
    train_loop(
        gcn, batch_features, batch_labels, loss_fn,
        max_epoch=max_epochs, learning_rate=learning_rate, is_classification=is_classification
    )
    print()
