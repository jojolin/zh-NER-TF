import logging, argparse

def cal_valid_words(sentence, merge_tags):
    '''
    calculate valid word
    '''
    words = []
    for pos, tag, st, ed in merge_tags:
        if (pos == 'E' and ed-st > 1) or (pos == 'S'):
            words.append((''.join(sentence[st:ed]), tag))

    return words

def merge_tag_(tags):
    '''
    tags: ['E_b', 'B_n', 'E_n', 'S_c', 'B_v', 'E_v', 'S_u', 'B_an', 'E_an']
    middle-result: [['E', 'b', 0, 1], ['E', 'n', 1, 3], ['S', 'c', 3, 4], ['E', 'v', 4, 6], ['S', 'u', 6, 7], ['E', 'an', 7, 9]]
    '''
    mres = []
    if len(tags) == 0:
        return []

    pos, tg = tags[0].split('_')
    mres.append([pos, tg, 0, 1])

    for i, tag in zip(range(2, len(tags)+1), tags[1:]):
        pos, tg = tag.split('_')
        pre = mres[-1]
        if pos not in ('B', 'S'):
            ppos = pre[0]
            ptg = pre[1]
            if tg == ptg and (ppos not in ('E', 'S')):
                # same tag and can merge, update pre
                mres[-1][0] = pos
                mres[-1][3] = i
                continue

        # otherwise, new start
        mres.append([pos, tg, pre[3], i])
    return mres

def merge_tag(tags):
    '''
    大成就和积累的丰富
    tags: ['E_b', 'B_n', 'E_n', 'S_c', 'B_v', 'E_v', 'S_u', 'B_an', 'E_an']
    return: [('b', 0, 1), ('n', 1, 3), ('c', 3, 4), ('v', 4, 6), ('u', 6, 7), ('an', 7, 9)]
    res: ['大/b', '成就/n', '和/c', '积累/v', '的/u', '丰富/an']

    '''
    mres = merge_tag_(tags)
    return [(x[1], x[2], x[3]) for x in mres]

def get_entity2(tags, sents):
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
            i = j
        else:
            pass

        j+=1
    return res

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
