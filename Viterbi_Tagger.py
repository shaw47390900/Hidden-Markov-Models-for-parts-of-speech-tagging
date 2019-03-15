from __future__ import division
from collections import Counter
from collections import defaultdict
import math

# parse training data
# returns two lists:
# wbrown = list where every element is a list of words of a particular sentence
# tbrown = list where every element is a list of tags of a particular sentence
def split_wordtags(brown_train):
    wbrown = [] # will be a list of lists of words
    tbrown = []

    for sentence in brown_train:
        words = []
        tags = []

        # add two start symbols (we're calculating trigrams) and one stop symbol to every sentence
        sentence = "*/* */* " + sentence + "STOP/STOP"

        # wordtags contains a list of words and their tags separated by /
        wordtags = sentence.split()

        for wordtag in wordtags:
            last_slash = wordtag.rindex('/')
            words.append(wordtag[:last_slash])
            tags.append(wordtag[last_slash+1:])

        wbrown.append(words)
        tbrown.append(tags)

    return wbrown, tbrown


# Removing infrequent words from the training data improves the performance of the algorithm
# calc_known takes the list of words from the training data and removes words that appear less than 6 times
def calc_known(wbrown):
    knownwords = []

    # flatten list
    words = [item for sublist in wbrown for item in sublist]

    counted_words = Counter(words)

    for word in counted_words:
        if counted_words[word] > 5:
            knownwords.append(word)

    return knownwords


# replace infrequent words with the tag '_RARE_'
def replace_rare(brown, knownwords):
    rare = []

    for sentence in brown:
        replaced = []
        for word in sentence:
            if word in knownwords:
                replaced.append(word)
            else:
                replaced.append("_RARE_")
        rare.append(replaced)
    return rare

def floored_log(count):
    if count == 0:
        return -1000
    else:
        return math.log(count, 2)

# Calculate POS trigram probability from list of tags in training data
# P(tag_i | (tag_i-2, tag_i-1)) = P(tag_i-2, tag_i-1, tag) / P(tag_i-2, tag_i-1)
#                               = Count(tag_i-2, tag_i-1, tag) / Count(tag_i-2, tag_i-1)
# Returns a python dictionary where the keys are tuples that represent the trigram
# and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    qvalues = {}
    bigram_count = Counter()
    trigram_count = Counter()

    for sentence in tbrown:

        #count bigrams
        for i in range(0, len(sentence) - 1):
            bigram_count[(sentence[i], sentence[i + 1])] += 1

        #count trigrams
        for i in range(0, len(sentence) - 2):
            trigram_count[(sentence[i], sentence[i + 1], sentence[i + 2])] += 1

    #calculate
    for item in trigram_count:
        bigram = item[:-1]
        token_count = bigram_count[bigram]
        qvalues[item] = floored_log(trigram_count[item] / token_count)

    return qvalues

# Calculates emission probabilities to be used in viterbi algorithm
# Returns a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag and the value is the log probability of that word/tag pair
def calc_emission(wbrown, tbrown):
    evalues = {}
    tuple_count = Counter()
    tag_count = Counter()

    # count word/tag tuples and count POS tags
    for words, tags in zip(wbrown, tbrown):
        for i in range(0, len(tags)):
            tag_count[tags[i]] += 1
            tuple_count[(words[i], tags[i])] += 1

    # calculate emission probabilities
    # P(word | tag) = P(word, tag) / P(tag) = Count(word/tag) / Count(tag)
    for tuple in tuple_count:
        evalues[tuple] = floored_log(tuple_count[tuple] / tag_count[tuple[1]])

    return evalues

# To easily sort through trigram probabilities in the algorithm we create a transition table.
# transition_table is dict of bigrams that map to a list of tuples where tuple = (bigram, transition probability)
# e.g. if trigram probability is "NOUN NOUN VERB" -> -12.342
# then transition_table[NOUN, NOUN] = [(NOUN, VERB), -12.342]
def create_ttable(p_trigrams):
    transition_table = defaultdict(list)

    for probability in p_trigrams:
        bigram = (probability[0], probability[1])
        transition_table[bigram].append(((probability[1], probability[2]), p_trigrams[probability]))

    return transition_table

# emission_table will be a dict where the key is the word and the value is a list of tuples (part of speech, p(POS|word)
def create_etable(emissions):
    emission_table = defaultdict(list)

    for emission in emissions:
        word = emission[0]
        emission_table[word].append((emission[1], emissions[emission]))

    return emission_table

# This function takes data to tag (brown), a list of known words (knownwords), trigram probabilities (qvalues)
# and emission probabilities (evalues) and implements the viterbi algorithm to output a list of tagged sentences
# Viterbi algorithm: multiply path probability by transition probability and emission probability
# keep track of most likely paths, then at the end of sentence find likeliest path through all words
# Initial path probability of 0.
def viterbi(brown, knownwords, qvalues, evalues):
    tagged = []

    transition_table = create_ttable(qvalues)
    emission_table = create_etable(evalues)

    for sentence in brown:
        tagged_sentence = ""

        #initialize 0 state
        path_probability = {('*', '*'): (0.0, [])}

        for word in sentence:
            if word == "*":
                continue
            if word not in knownwords:
                word = "_RARE_"

            # This implementation assumes that each trigram has been seen by the training data. this is not always true,
            # so we need a fallback method. The fallback used here is to just propagate it through and let the emission
            # probability take care of sorting the paths. In order to do that, we need to save a copy of the
            # path_probability dictionary so we can use it later if necessary.
            path_copy = path_probability.copy()

            output_matrix = defaultdict(list)  #output_matrix[next_state] = list of (current path, next_state, probability)
            for path in path_probability:
                current_path = path_probability[path]
                if path in transition_table.keys():
                    output_tuples = transition_table[path] #output_tuples are a list of tuples where tuple = (bigram, probability)
                else:
                    continue
                for next_state in output_tuples:  #next_state is tuple = (bigram, probability)
                    probability = current_path[0] + next_state[1]
                    output_matrix[next_state[0]].append((current_path[1], next_state[0], probability))

            # We have the output matrix, now we need to add probabilities for each tuple
            # and find the max probability for each next state
            max_probabilities = []
            for current_state in output_matrix:
                max_probability = ([], ("",""), -10000.0)
                for state in output_matrix[current_state]:
                    if state[2] > max_probability[2]:
                        max_probability = state
                max_probabilities.append(max_probability)

            path_probability = {} #clear out old path_probability vector

            # Multiply max probabilities with emission table to get next path_probability
            emissions = emission_table[word]

            for probable_state in max_probabilities:  #probable_state is a tuple ([current_path, next_state, prob)
                next_tag = probable_state[1][1]
                current_path = probable_state[0][:]
                path_prob = probable_state[2]
                for emission in emissions:  #emission is a tuple (POS, prob)
                    if emission[0] == next_tag:
                        current_path.append(probable_state[1])
                        emission_probability = path_prob + emission[1]
                        # path_probability[current_state] = (probability, path)
                        path_probability[probable_state[1]] = (emission_probability, current_path)

            #fallback method: assume every path is the best path, move forward in sentence with the emissions tacked onto path
            if not path_probability:
                for path in path_copy:
                    current_path = path_copy[path][1]
                    previous_tag = path[1]
                    current_prob = path_copy[path][0]
                    for emission in emissions: #(POS, prob)
                        next_state = (previous_tag, emission[0])
                        current_path.append((next_state))
                        path_prob = current_prob + emission[1]
                        path_probability[next_state] = (path_prob, current_path)

        best_path = (-10000, [])
        for path in path_probability:
            tuple=path_probability[path]
            if tuple[0] > best_path[0]:
                best_path = tuple
        i=0
        for word in sentence:
            if word == "*" or word == "STOP":
                continue
            tagged_sentence += word + "/" + best_path[1][i][1] + " "
            i+= 1
        tagged_sentence += "\n"
        tagged.append(tagged_sentence)
    return tagged

def score(tagged_sentences, reference_file):
    infile = open(reference_file, "r")
    correct_sentences = infile.readlines()
    infile.close()

    num_correct = 0
    total = 0

    for tagged_sent, correct_sent in zip(tagged_sentences, correct_sentences):
        user_tok = tagged_sent.split()
        correct_tok = correct_sent.split()

        if len(user_tok) != len(correct_tok):
            continue

        for u, c in zip(user_tok, correct_tok):
            if u == c:
                num_correct += 1
            total += 1

    score = float(num_correct) / total * 100

    print("Percent correct tags:", score)


# output trigram probabilities
def trigram_output(qvalues):
    outfile = open("trigrams.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

# output emission probabilities
def emission_output(evalues):
    outfile = open("emissions.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()

# output tagged sentences
def tagged_output(tagged):
    outfile = open('tagged_sentences.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

def main():
    # open Brown training data
    infile = open("brown.train.tagged.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols
    wbrown, tbrown = split_wordtags(brown_train)

    # calculate trigram probabilities
    qvalues = calc_trigrams(tbrown)
    trigram_output(qvalues)

    # calculate list of words with count > 5
    knownwords = calc_known(wbrown)

    # get a version of wbrown with rare words replaced with '_RARE_'
    wbrown_rare = replace_rare(wbrown, knownwords)

    # calculate emission probabilities
    evalues = calc_emission(wbrown_rare, tbrown)
    emission_output(evalues)

    # delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data
    infile = open("brown.test.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    temp_brown = []
    #format Brown development data
    for sentence in brown_dev:
        add = "* * " + sentence + " STOP"
        tokens = add.split()
        temp_brown.append(tokens)

    brown_dev = temp_brown

    #do viterbi on brown_dev
    viterbi_tagged = viterbi(brown_dev, knownwords, qvalues, evalues)
    tagged_output(viterbi_tagged)

    #print score
    score(viterbi_tagged, "brown.test.tagged.txt")

if __name__ == "__main__": main()
