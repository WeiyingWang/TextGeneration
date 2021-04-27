import os
import json
import pickle


def load_images(data_path):
    image_path = os.path.join(data_path, 'full_img.pkl')
    datas = pickle.load(open(image_path, 'rb'))
    images = {}
    for key in datas:
        images[key] = datas[key]
    print('Total images ', len(images))
    return images

def load_vocabs(data_path):
    dict_path = os.path.join(data_path, 'dict.json')
    word2id = json.load(open(dict_path, 'r', encoding='utf-8'))
    id2word = {word2id[key]:key for key in word2id}
    print('Vocabulary Size', len(word2id))
    return word2id, id2word