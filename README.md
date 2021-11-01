# Capstone Project Checkpoint

## Models

The checkpoint code section compares two baseline models:

- FCN: fully connected neural network
- GCN: graph convolutional neural network

Both models have only one hidden layer with the same dimension by design.

## Dataset

Two dataset are used to measure respective performances:

- cora citation network: a **classification** task to predict the category of papers;
- finefoods reviews: a **regression** task to predict review scores users give to products.

## File Digestion

The main file, [run.py](run.py), read in one of the above dataset and run both FCN and GCN on it, and report accuracy of respective models after each epoch.

This project also contains three helper files:

- [data_reader.py](src/data_reader.py): read in data. To take in new dataset, we shall implement methods in this file;
- [data_preprocessor.py](src/data_preprocessor.py): build a graph on the raw feature matrix, if graph information is not available;
- [baseline_models.py](src/baseline_models.py): contains two baseline models described above.


## Quick Start

It is easy to run this project. There is no need to change anything except [config.ini](config.ini). Note that the very first to to run takes a bit longer since we are downloading and unzipping the dataset.

- To test on cora citation, use the following parameters:

    - source = cora
    - convert_to_undirected = False (or True, if a symmetrized version is preferred)
    - model_type = classification
    - max_epoch = 100
    - hidden_layer_dim = 64
    - learning_rate = 1
    - build_graph = False (if interested in an alternative graph, could set this to True)
    - graph_builder = nn_graph_builder (either one is fine)

    and then run [run.py](run.py).

- To test on fienfoods instead, use the following parameters:

    - source = finefoods
    - convert_to_undirected = True (does not matter)
    - model_type = regression
    - max_epoch = 100
    - hidden_layer_dim = 64
    - learning_rate = 0.1
    - build_graph = True (need to build a graph)
    - graph_builder = nn_graph_builder (this one is recommended for obtaining quicker results)

    and then run [run.py](run.py). Note that running on finefoods take a little longer. The majority of time is on building the graph.

Please check out other details in each file.
