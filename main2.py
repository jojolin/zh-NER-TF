import json
import tensorflow as tf
import numpy as np
import os, time
from model import BiLSTM_CRF
from utils import str2bool, get_logger, merge_tag
from data import read_corpus, read_dictionary, random_embedding


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

class Args():
    def __init__(self):

        self.train_data='rb1' # '测试集，校验集，测试集的文件位置'

        self.mode='train' # '步骤：build_tag, train, test, demo'

        self.model = str(int(time.time()))
        # 保存的模型路径 (self.train_data)_save/self.model
        self.output_path = os.path.join('.', self.train_data+"_save", self.model)

        self.sentence='' # 分词句子
        self.sfile='' # 分词文件

        self.batch_size=128 #64 # help='#sample of each minibatch'
        self.epoch=5 #40  # help='#epoch of training'
        self.hidden_dim=300 # help='#dim of hidden state'
        self.optimizer='Adam' # help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD'
        self.CRF=True # help='use CRF at the top layer. if False, use Softmax'
        self.lr=0.001 # help='learning rate'
        self.clip=5.0 # help='gradient clipping'
        self.dropout=0.5 # help='dropout keep_prob'
        self.update_embedding=True # help='update embedding during training'
        self.pretrain_embedding='random' # help='use pretrained char embedding or init it randomly'
        self.embedding_dim=300 # help='random init char embedding_dim'
        self.shuffle=True # help='shuffle training data before each epoch'

    def __str__(self):
        st = ''
        for prop in dir(self):
            if prop.startswith('_'):
                continue
            else:
                st +='prop: %s, value: %s' % (prop,  args.__getattribute__(prop))+'\n'
        return st

args = Args()

def load_stuff():
    ## get char embeddings
    word2id = read_dictionary(os.path.join('.', 'word2id.pkl'))

    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')
    return word2id, embeddings

def init_paths_config():
    ## paths setting
    paths = {}
    output_path = args.output_path
    if not os.path.exists(output_path): os.makedirs(output_path)

    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    paths['summary_path'] = summary_path

    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    paths['model_path'] = model_path

    ckpt_prefix = os.path.join(model_path, "model")
    paths['ckpt_prefix'] = ckpt_prefix

    result_path = os.path.join(output_path, "results")
    if not os.path.exists(result_path): os.makedirs(result_path)
    paths['result_path'] = result_path

    log_path = os.path.join(result_path, "log.txt")
    get_logger(log_path).info(str(args))
    paths['log_path'] = log_path

    return paths

def build_tags(data):
    tagu = []
    for d in data:
        for sents, tags in d:
            for tag in tags:
                if not tag in tagu:
                    tagu.append(tag)

    tagu.insert(0, '_')    # add empty tag
    return tagu

def load_taglabel(filep):
    import json
    tag_label = {}
    with open(filep, 'r') as r:
        tag_label = json.loads(r.readline())
    print('loading tag_label: %s' % tag_label)
    return tag_label

def build_tag():
    train_path = os.path.join('.', args.train_data, 'train.txt')
    train_data = read_corpus(train_path)
    dev_path = os.path.join('.', args.train_data, 'dev.txt')
    dev_data = read_corpus(dev_path)
    test_path = os.path.join('.', args.train_data, 'test.txt')
    test_data = read_corpus(test_path)
    tags = build_tags([train_data, dev_data, test_data])
    print('tags: %s' % tags)
    tag_label = {}
    for i, t in enumerate(tags):
        tag_label[t] = i
    with open(os.path.join('.', args.train_data, 'tags.txt'), 'w') as w:
        w.write(json.dumps(tag_label))
        print('save tag_label: %s' % tag_label)

def train(word2id, embeddings, paths):
    train_path = os.path.join('.', args.train_data, 'train.txt')
    train_data = read_corpus(train_path)
    dev_path = os.path.join('.', args.train_data, 'dev.txt')
    dev_data = read_corpus(dev_path)
    # test_path = os.path.join('.', args.train_data, 'test.txt')
    # test_data = read_corpus(test_path)
    print("train data: {0}\ndev data: {1}".format( len(train_data), len(dev_data)))

    taglabel = load_taglabel(os.path.join('.', args.train_data, 'tags.txt'))
    model = BiLSTM_CRF(args, embeddings, taglabel, word2id, paths, config=config)
    model.build_graph()
    model.train(train=train_data, dev=dev_data)

def test(word2id, embeddings, paths):
    '''模型评估'''
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    paths['model_path'] = ckpt_file

    test_path = os.path.join('.', args.train_data, 'test.txt')
    test_data = read_corpus(test_path)
    test_size = len(test_data)
    print("test data: {}".format(test_size))

    taglabel = load_taglabel(os.path.join('.', args.train_data, 'tags.txt'))
    model = BiLSTM_CRF(args, embeddings, taglabel, word2id, paths, config=config)
    model.build_graph()
    model.test(test_data)

def demo(word2id, embeddings, paths, mode='demo'):
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    paths['model_path'] = ckpt_file

    taglabel = load_taglabel(os.path.join('.', args.train_data, 'tags.txt'))
    model = BiLSTM_CRF(args, embeddings, taglabel, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()

    def handle_sen(sentence):
        demo_sent = list(sentence.strip())
        demo_data = [(demo_sent, ['_'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        # res = get_entity(tag, demo_sent)
        tag_merge = merge_tag(tag)
        print(tag_merge)
        res = []
        for t, st, ed in tag_merge:
            # [(tag, start_index, end_index)]
            res.append(''.join(demo_sent[st:ed]) + '/' + t)
        return ''.join(demo_sent), tag, res

    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        if mode == 'demoi':
            while(1):
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace():
                    print('See you next time!')
                    break
                else:
                    sent, tag, res = handle_sen(demo_sent.strip())
                    print('sen: %s\ntag: %s\nres: %s' % (sent, tag, res))
        else:
            if args.sentence != '':
                demo_sent = args.sentence
                sent, tag, res = handle_sen(demo_sent.strip())
                print('sen: %s\ntag: %s\nres: %s' % (sent, tag, res))
            elif args.sfile != '':
                with open(args.sfile, 'r') as r:
                    lines = r.readlines()
                    for line in lines:
                        sent, tag, res = handle_sen(line.strip())
                        print('sen: %s\ntag: %s\nres: %s' % (sent, tag, res))

def run():
    if args.mode == 'build_tag':
        # build tag
        build_tag()
        return

    word2id, embeddings = load_stuff()
    paths = init_paths_config()
    if args.mode == 'train':
        train(word2id, embeddings, paths)

    elif args.mode == 'test':
        test(word2id, embeddings, paths)

    elif args.mode == 'demoi' or args.mode == 'demo':
        if args.mode == 'demo' and args.sentence == '' and args.sfile == '':
            print('empty sentence! please specify sentence with -s `sentence` or -f `file-contains-sentences` ')
            return

        demo(word2id, embeddings, paths, args.mode)
    else:
        pass

def main():
    # 每一个新的训练集必须先运行以下命令
    args.mode = 'build_tag'
    args.train_data = "rb1"
    args.model = str(int(time.time()))

    # train
    args.mode = 'train'
    args.train_data = "rb1"
    args.output_path= '1234567891011'
    #args.model = '123'

    # test
    args.mode = 'test'
    args.train_data = "rb1"
    args.model = '1234567'

    # demo
    args.mode = 'demo'
    args.train_data = "rb1"
    args.output_path= '1234567891011'
    args.sentence = '基于服务资源的需求检索'

    args.mode='demo'
    run()

if __name__ == '__main__':
    main()
