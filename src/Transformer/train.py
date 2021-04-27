'''
    Hierachical Graph Network
'''

import os
import sys
import torch
import json
import time
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.optim as Optim
from torch.autograd import Variable

import utils
import dataset
import modules
from eval import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser(description='train.py')
# hyperparameter
parser.add_argument('-n_embs', type=int, default=512, help='Embedding size')
parser.add_argument('-img_embs', type=int, default=2048, help='Image embedding size')
parser.add_argument('-dim_ff', type=int, default=512, help='Feed forward hidden size')
parser.add_argument('-n_head', type=int, default=8, help='head number')
parser.add_argument('-n_block', type=int, default=4, help='block number')

parser.add_argument('-batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('-epoch', type=int, default=100, help='Number of Epoch')
parser.add_argument('-report', type=int, default=50, help='Report interval')
parser.add_argument('-dropout', type=float, default=0.5, help='Dropout Rate')
parser.add_argument('-max_len', type=int, default=40, help='Max length of captions')
parser.add_argument('-lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('-beam_size', type=int, default=1, help='Beam Size')

parser.add_argument('-data_path', type=str, default='../../../../data/ImageEdit/')
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-valid_type', type=str, default='cider', help='Validation method')
parser.add_argument('-out_path', type=str, default='./checkpoint', help='out path')
parser.add_argument('-out_file', type=str, default=None, help = 'result for evaluation')
parser.add_argument('-mode', type=str, default=None)
args = parser.parse_args()

# random seed
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if args.valid_type is None:
    print('Please enter validation method ')
    sys.exit()

# load images
images = utils.load_images(args.data_path)
vocabs, rev_vocabs = utils.load_vocabs(args.data_path)
vocab_size = len(vocabs)

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)

def get_dataset(data_path, images, set_type=None, vocabs=vocabs, is_train=True):
    return dataset.Dataset(data_path,
                           vocabs = vocabs,
                           rev_vocabs = rev_vocabs,
                           images = images,
                           max_len = args.max_len,
                           set_type = set_type)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

def train():
    train_set = get_dataset(os.path.join(args.data_path, 'train.json'), images, None)
    valid_set = get_dataset(os.path.join(args.data_path, 'val.json'), images, args.valid_type)

    train_batch = get_dataloader(train_set, args.batch_size, is_train=True)

    model = modules.Model(n_embs = args.n_embs,
                          img_embs = args.img_embs,
                          n_hidden = args.dim_ff,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = train_set.CLS)

    #model = nn.DataParallel(model)
    if args.restore != '':
        model_dict = torch.load(args.restore)
        model.load_state_dict(model_dict)

    model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) #, weight_decay=0.01)
    # scheduler = Optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9)
    best_score = -100000000
    no_beat = 0
    for i in range(args.epoch):
        model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        count = 0
        total = len(train_set) // args.batch_size + 1

        for batch in train_batch:
            model.zero_grad()
            Img1, Img2, Des = batch
            Img1 = Variable(Img1).cuda()
            Img2 = Variable(Img2).cuda()
            Des = Variable(Des).cuda()

            loss = torch.sum(model(Img1, Img2, Des))

            loss.backward()
            optim.step()

            report_loss += loss.data.item()
            n_samples += len(Img1.data)
            count += 1

            if count % args.report == 0 or count == total:
                # report loss
                print('%d/%d, epoch: %d, report_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, time.time() - start_time))
                if i > 1:
                    if args.valid_type == 'P':
                        score = eval_p(valid_set, model)
                    else:
                        score = eval(valid_set, model)
                    model.train()

                    if score > best_score:
                        no_beat = 0
                        best_score = score
                        print('Score Beat ', score, '\n')
                        save_model(os.path.join(args.out_path, 'best_checkpoint.pt'), model)
                    else:
                        no_beat += 1
                        save_model(os.path.join(args.out_path, 'checkpoint.pt'), model)
                        print('Term ', no_beat, 'Best_score', best_score, '\n')
                        if no_beat == 20:
                            sys.exit()
                    report_loss, start_time, n_samples = 0, time.time(), 0
        #scheduler.step()
        print('Learning Rate ', optim.state_dict()['param_groups'][0]['lr'])
    return 0

def eval_p(dev_set, model):
    print('Starting Evaluation...')
    start_time = time.time()
    model.eval()
    dev_batch = get_dataloader(dev_set, args.batch_size, is_train=False)
    loss = 0
    with torch.no_grad():
        for batch in dev_batch:
            Img1, Img2, Des = batch
            Img1 = Variable(Img1).cuda()
            Img2 = Variable(Img2).cuda()
            Des = Variable(Des).cuda()

            loss += torch.sum(model(Img1, Img2, Des)).item()
    print('Loss: ', loss / len(dev_set))
    print("evaluting time:", time.time() - start_time)
    return -loss / len(dev_set)

def eval(dev_set, model):
    print('Starting Evaluation...')
    start_time = time.time()
    model.eval()
    loss = 0
    candidates = {}
    references = {}
    with torch.no_grad():
        for i in range(len(dev_set)):
            ImgID, Img1, Img2, Caps = dev_set.get_data(i)

            Img1 = Variable(Img1).cuda()
            Img2 = Variable(Img2).cuda()

            ids = model.generate(Img1, Img2, beam_size=args.beam_size).data.tolist()

            sentences = transform(ids)
            if i < 6:
                print('   ', ' '.join(sentences))

            candidates[ImgID] = [' '.join(sentences)]
            references[ImgID] = [s for s in Caps]

    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    print('CIDEr: ', score)
    print("evaluting time:", time.time() - start_time)
    return score

def test():
    test_set = get_dataset(os.path.join(args.data_path, 'test.json'), images)

    model = modules.Model(n_embs = args.n_embs,
                          img_embs = args.img_embs,
                          n_hidden = args.dim_ff,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          vocab_size = vocab_size,
                          dropout = args.dropout,
                          max_len = args.max_len,
                          CLS = test_set.CLS)

    model_dict = torch.load(args.restore)
    model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    model.cuda()
    model.eval()

    with torch.no_grad():
        with open(args.out_file, 'w', encoding='utf-8') as fout:
            for i in range(len(test_set)):
                ImgID, Img1, Img2, Caps = test_set.get_data(i)
                Img1 = Variable(Img1).cuda()
                Img2 = Variable(Img2).cuda()

                ids = model.generate(Img1, Img2, beam_size=args.beam_size).data.tolist()
                sentences = transform(ids)

                sample = {'ImgId': ImgID,
                          'candidates': sentences,
                          'references': [s.split(' ') for s in Caps]}
                jterm = json.dumps(sample, ensure_ascii=False)
                fout.write(jterm + '\n')
            

def transform(ids):
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[wid])
    return sentences


if __name__ == '__main__':
    if args.mode == 'train':
        print('------Train Mode------')
        train()
    elif args.mode == 'test':
        print('------Test  Mode------')
        test()
    else:
        print('Please enter the mode')
