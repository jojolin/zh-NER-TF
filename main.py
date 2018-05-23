import tensorflow as tf
import numpy as np
import os, argparse, time
from model import BiLSTM_CRF
from utils import str2bool, get_logger
from data import read_corpus, read_dictionary, random_embedding


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', '-t', type=str, default='data_path', help='train data source')
# parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', '-m', type=str, default='demo', help='train/test/demo')
parser.add_argument('--model', type=str, default='1521112368', help='model for test and demo')
parser.add_argument('--sentence', '-s', type=str, default='', help='sentence to tag')
args = parser.parse_args()

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
    timestamp = str(int(time.time())) if args.mode == 'train' else args.model

    output_path = os.path.join('.', args.train_data+"_save", timestamp)
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

paths = init_paths_config()

def get_entity(tags, sents):
    '''
    e.g: ['B_n', 'E_n', 'B_v', 'E_v', 'B_v', 'E_v', 'B_a', 'E_a', 'B_v', 'E_v', 'S_w', 'S_d', 'B_v', 'E_v', 'S_u', 'S_v'] ->
    '''
    i = j = 0
    res = []
    while j < len(tags):
        jpos = tags[j].split('_')[0]
        tag = tags[j].split('_')[1]
        if jpos == 'B':
            # word BEGIN
            i = j
        elif jpos == 'S':
            # world SINGLE
            i = j
            res.append('%s/%s' % (''.join(sents[i:j+1]), tag))
        elif jpos == 'E':
            # word END
            res.append('%s/%s' % (''.join(sents[i:j+1]), tag))

        j+=1
    return res

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
    import json
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
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['_'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    res = get_entity(tag, demo_sent)
                    print('sen: %s\ntag: %s\nres: %s' % (demo_sent, tag, res))
        else:
            demo_sent = args.sentence
            demo_sent = list(demo_sent.strip())
            demo_data = [(demo_sent, ['_'] * len(demo_sent))]
            tag = model.demo_one(sess, demo_data)
            res = get_entity(tag, demo_sent)
            print('sen: %s\ntag: %s\nres: %s' % (demo_sent, tag, res))

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
        if args.mode == 'demo' and args.sentence == '':
            print('empty sentence! please specify sentence with -s `sentence`')
            return

        demo(word2id, embeddings, paths, args.mode)
    else:
        pass

def main():
    run()

if __name__ == '__main__':
    main()
