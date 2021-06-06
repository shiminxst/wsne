WSNE
=====

This repository implements the Service-GCN for Web service network embedding.

## Data

two datasets were used:

- Webservice: The Mashup and API services together with their composition links, crawled from ProgrammableWeb
- Wiki: The Web page networks between Web pages and their hyperlinks.

## Requirement

  * Python 3.6
  * Tensorflow 1.14.0

## Usage

The wsne_1layer.py implements a 1-layer Service-GCN, and wsne_2layer.py implements a 2-layer Service-GCN.

To run the classification task:
```sh
python wsne/wsne_1layer.py 
```

To run the clustering task:
```sh
python wsne/service_clustering.py
```
## Reference

[1] Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907. 2016 Sep 9.
