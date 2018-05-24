import statis

def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    print(label_predict)

    # eval_perl = "./conlleval_rev.pl"
    # os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    # with open(metric_path) as fr:
    #    metrics = [li.strip() for li in fr]

    result = statis.st(label_path, metric_path)
    return result.split('\n')

