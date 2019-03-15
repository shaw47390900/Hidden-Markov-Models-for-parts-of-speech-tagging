import numpy as np
import string

def prediction(sorted_tag, test_raw):
    fo = open("test_tagged.txt", 'w')

    for line in test_raw:
        word_lst = line.strip().split(' ')

        for word in word_lst:
            fo.write(word)
            if word.lower() in sorted_tag.keys():
                fo.write('/'+sorted_tag[word.lower()][0][0]+' ')

            else:
                if word.istitle():
                    fo.write('/nnp ')
                elif  word.endswith('s'):
                    fo.write('/nns ')
                elif word[0].isdigit():
                    fo.write('/cd ')
                elif '-' in word:
                    fo.write('/jj ')
                elif  word.endswith('ing'):
                    fo.write('/vbg ')
                elif  word in string.punctuation:
                    fo.write('/. ')
                else:
                    fo.write('/nn ')
        fo.write('\n')
    fo.close()
    print(accuracy('test_tagged.txt', 'brown.test.tagged.txt'))


def make_dict(content):
    word_dict = {}
    word_corpus = []

    for line in content:
        raw_lst = line.lower().strip().split(' ')

        for item in raw_lst:
            last_slash = item.rindex('/')
            word = item[:last_slash]
            tag = item[last_slash+1:]
            if word not in word_corpus:
                word_corpus.append(word)
                word_dict[word] = {}
            if tag in word_dict[word]:
                word_dict[word][tag] += 1
            else:
                word_dict[word][tag] = 1

    res_dict = {}
    for key, value in word_dict.items():
        res_dict[key] = sorted(value.items(), key=lambda item: item[1], reverse=True)
        res_dict[key] = np.array(res_dict[key])
    return word_corpus, res_dict

def accuracy(prediction_file, tagged_file):
    taggedfile = open(tagged_file, "r")
    tagged_sentences = taggedfile.readlines()
    taggedfile.close()

    predictionfile = open(prediction_file, "r")
    prediction_sentences = predictionfile.readlines()
    predictionfile.close()

    accuracy = 0
    total = 0

    for tagged_sent, prediction_sent in zip(tagged_sentences, prediction_sentences):
        tagged_tok = tagged_sent.split()
        prediction_tok = prediction_sent.split()
        if len(tagged_tok) != len(prediction_tok):
            continue
        for u, c in zip(tagged_tok, prediction_tok):
            if u == c:
                accuracy += 1
            total += 1

    score = float(accuracy) / total * 100
    print("Percent correct tags:", score)

trainfile = open("brown.train.tagged.txt", "r")
train_tagged = trainfile.readlines()
trainfile.close()
testfile = open("brown.test.txt", "r")
test_raw = testfile.readlines()
testfile.close()
word_corpus, tag_dict = make_dict(train_tagged)
prediction(tag_dict, test_raw)
