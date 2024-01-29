'''根据类别映射找到对应的类别名称'''
import json
import os

import requests
import torch


def get_classes_with_pred(preds, top=1):
    """Decode the prediction of an ImageNet model

    # Arguments
        preds: torch tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return

    # Return
        lists of top class prediction classes and id
    """

    class_index_path = 'https://s3.amazonaws.com\
    /deep-learning-models/image-models/imagenet_class_index.json'

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`get_classes_with_pred` expects a batch of predciton'
        '(i.e. a 2D array of shape (samples, 1000)).'
        'Found array with shape: ' + str(preds.shape)
        )

    if not os.path.exists('./data/imagenet_class_index.json'):
        r = requests.get(class_index_path).content
        with open('./data/imagenet_class_index.json', 'w+') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)

    _, top_indices = torch.topk(preds, top)
    predicted_classes = [class_index_dict[str(i.item())][1] for i in top_indices]
    predicted_id = [j.item() for j in top_indices]
   
    return predicted_classes, predicted_id

def get_classes_with_index(index):
    '''根据预测类别的index，返回类别名称'''
    class_index_path = 'https://s3.amazonaws.com\
    /deep-learning-models/image-models/imagenet_class_index.json'

    if not os.path.exists('./data/imagenet_class_index.json'):
        r = requests.get(class_index_path).content
        with open('./data/imagenet_class_index.json', 'w+') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)
    
    predicted_classes = [class_index_dict[str(i.item())][1] for i in index]

    return predicted_classes
    