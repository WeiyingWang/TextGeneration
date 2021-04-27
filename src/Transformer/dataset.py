import json
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, images, vocabs, rev_vocabs, max_len, set_type):
        # set parameter
        self.images = images
        self.max_len = max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.set_type = set_type

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.PAD = vocabs['<PAD>']
        self.UNK = vocabs['<UNK>']
        self.CLS = vocabs['<CLS>']

        # load data
        if self.set_type == 'P':
            self.load_data_multi_sents(data_path)
        else:
            self.load_data(data_path)

    def load_data_multi_sents(self, data_path):
        print('Validation with generation probability')
        self.datas = []
        dropdata = 0
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                if jterm['img1'] in self.images and jterm['img2'] in self.images:
                    for s in jterm:
                        sample = {'img1': jterm['img1'],
                                  'img2': jterm['img2'],
                                  'sentences': s}
                        self.datas.append(sample)
                else:
                    dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')

    def load_data(self, data_path):
        self.datas = []
        dropdata = 0
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                if jterm['img1'] in self.images and jterm['img2'] in self.images:
                    self.datas.append(jterm)
                else:
                    dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        img1 = torch.Tensor(self.images[data['img1']])
        img2 = torch.Tensor(self.images[data['img2']])
        cap, cap_len = self.padding(data['sentences'].split(' '))
        return img1, img2, cap

    def get_data(self, index):
        # test mode
        data = self.datas[index]
        img1 = torch.Tensor(self.images[data['img1']])
        img2 = torch.Tensor(self.images[data['img2']])
        ImgId = data['img1']+'_'+data['img2']
        return ImgId, img1, img2, self.datas[index]['sentences']

    def padding(self, sent):
        if len(sent) > self.max_len-3:
            sent = sent[:self.max_len-3]
        text = list(map(lambda t: self.vocabs.get(t, self.UNK), sent))
        # text = [self.CLS, self.BOS] + text + [self.EOS, self.PAD]
        text = [self.BOS] + text + [self.EOS]
        length = len(text)
        T = torch.cat([torch.LongTensor(text), torch.zeros(self.max_len - length).long()])
        return T, length

    def CLS(self):
        return self.CLS