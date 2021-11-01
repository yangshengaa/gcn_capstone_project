""" 
download data from online sources 
"""

# import packages
import gzip
import os 
import tarfile
import urllib.request

DATAPATH = 'data/'

# ==========================
# ------ download all ------
# ==========================

def download_all_data():
    """ download all data """
    if not os.path.exists(os.path.join(DATAPATH, 'cora')):
        print('Downloading Cora Dataset')
        download_cora()
    else:
        print('Cora Data Downloaded')
    if not os.path.exists(os.path.join(DATAPATH, 'finefoods')):
        print('Downloading Finefoods Dataset')
        download_finefoods()
    else:
        print('FineFoods Data Downloaded')

# ==========================
# ----- specific dataset ---
# ==========================

def download_cora():
    """ download cora dataset """
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    cora_tgz_datapath = os.path.join(DATAPATH, 'cora.tgz')
    urllib.request.urlretrieve(cora_url, cora_tgz_datapath)
    # unpack 
    file = tarfile.open(cora_tgz_datapath)
    file.extractall(DATAPATH)
    file.close()
    # delete tgz file
    os.remove(cora_tgz_datapath)

def download_finefoods():
    """ download finefoods dataset """
    finefoods_url = 'https://snap.stanford.edu/data/finefoods.txt.gz'
    finefoods_gz_datapath = os.path.join(DATAPATH, 'finefoods.txt.gz')
    urllib.request.urlretrieve(finefoods_url, finefoods_gz_datapath)
    # unpack
    dir_name = os.path.join(DATAPATH, 'finefoods')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    txt_output = open(os.path.join(dir_name, 'finefoods.txt'), 'wb')
    with gzip.open(finefoods_gz_datapath, 'rb') as f:
        for line in f:
            txt_output.write(line)
    txt_output.close()
    # delete tgz file
    os.remove(finefoods_gz_datapath)


# for testing purposes
if __name__ == '__main__':
    DATAPATH = '../' + DATAPATH
    download_all_data()
