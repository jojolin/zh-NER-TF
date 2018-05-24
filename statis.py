# coding: utf8
import sys
import utils

def st(label_path, save_path=''):
    correct = 0
    total = 0
    sentence = []
    ori_tags = []
    label_tags = []
    with open(label_path, 'r') as r:
        lines = r.readlines()
        for line in lines:
            if line.strip() == '':
                continue

            total += 1
            char, ori_tag, label_tag = line.strip().split(' ')
            sentence.append(char)
            ori_tags.append(ori_tag)
            label_tags.append(label_tag)
            if ori_tag == label_tag:
                correct += 1

    ori_tags_merge_ = utils.merge_tag_(ori_tags)
    label_tags_merge_ = utils.merge_tag_(label_tags)
    # 原来有效单词
    ori_valid_words = utils.cal_valid_words(sentence, ori_tags_merge_)
    # 标记的有效单词
    label_valid_words = utils.cal_valid_words(sentence, label_tags_merge_)
    vc = 0
    for lvw in label_valid_words:
        if lvw in ori_valid_words:
            vc += 1

    tp = vc
    fp = len(label_valid_words) - tp
    fn = len(ori_valid_words) - vc
    #tn = 0
    precision = tp * 1.0 / (tp + fp)  # len(label_valid_words)
    recall = tp * 1.0 / (tp + fn)    # len(ori_valid_words)
    fb1 = 2 * precision  * recall / (precision + recall)
    accuracy = correct * 100.0 / total
    # processed 796 tokens with 0 phrases; found: 0 phrases; correct: 0.
    # accuracy:  28.02%; precision:   0.00%; recall:   0.00%; FB1:   0.00
    resfmt = 'tokens:%s, tag correct: %s, tag accuracy: %.2f\n' + \
             'phrases: %s, found phrases: %s, correct phrases: %s\n' + \
             'phrases precision: %.2f;  phrases recall: %.2f; FB1: %.2f'
    result = resfmt % (total, correct, accuracy,
            len(ori_valid_words), len(label_valid_words), vc,
            precision, recall, fb1)
    if not save_path == '':
        with open(save_path, 'w') as w:
            w.write(result)
    else:
        print(result)

    return result

def main():
    fp = sys.argv[1]
    st(fp)

if __name__ == '__main__':
    main()
