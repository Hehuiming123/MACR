import ml_collections


def get_b16_config():
    """
    about model
    :return:
    """
    c = ml_collections.ConfigDict()
    c.patches = ml_collections.ConfigDict({'size': (16, 16)})
    c.split = 'non-overlap'
    c.slide_step = 12
    c.hidden_size = 768
    c.transformer = ml_collections.ConfigDict()
    c.transformer.mlp_dim = 3072
    c.transformer.num_heads = 12
    c.transformer.num_layers = 12
    c.transformer.attention_dropout_rate = 0.0
    c.transformer.dropout_rate = 0.1
    c.classifier = 'token'
    c.representation_size = None
    

    c.eta = 0.2
    c.p = 4
    return c


config = get_b16_config()
vit_pretrain = '/root/siton-data-liangzhuominData/wz/TransFG/pretrained_vit/ViT-B_16.npz'  # pretrained weights for backbone, please edit it for your configuration
epochs = 60
lr = 1e-4
  
zoom_size = 512
input_size = 448 
batch_size = 10
which_set = 'nabirds'
if which_set == 'cub':
    train_csv = 'datasets/cub/train.csv'
    test_csv = 'datasets/cub/test.csv'
    train_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/CUB_200_2011/CUB_200_2011/images/'
    test_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/CUB_200_2011/CUB_200_2011/images/'
    config.num_classes = 200
elif which_set == 'dog':
    train_csv = 'datasets/dog/train.csv'
    test_csv = 'datasets/dog/test.csv'
    train_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/dog/Images/'
    test_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/dog/Images/'
    config.num_classes = 120
elif which_set == 'nabirds':
    train_csv = 'datasets/nabirds/train.csv'
    test_csv = 'datasets/nabirds/test.csv'
    train_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/nabirds/images/'
    test_root = '/root/siton-data-liangzhuominData/wz/TransFG/dataset/nabirds/images/'
    config.num_classes = 555
else:
    assert False, 'no dataset'

if which_set == 'dog':
    lr_ml = 100
    alpha = 0.01
else:
    lr_ml = 1
    alpha = 1

beta = [1, 1, 1, 1]
if which_set == 'cub':
    beta = [0.2, 0.2, 0.8, 0.8]


CUDA_VISIBLE_DEVICES = '0'    # '0,1,2,3'
momentum = 0.9
weight_decay = 5e-4
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

weight_path = 'checkpoints/nabirds/nabirds2-44.pth'  # weight for val.py
