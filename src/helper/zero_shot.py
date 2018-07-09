import json
import helper.word_embeddings
from collections import defaultdict, OrderedDict

class Zero_Shooter:

    def __init__(self, modelpath, names_list):
        with open(modelpath + 'all_highest_probs.json', 'r') as f:
            self.candidates = json.load(f)
        self.word_that_are_names = names_list

    def get_predictions(self, region_id):
        predictions = list()
        tmp_dict = self.candidates[region_id]
        sorted_tmp = OrderedDict(sorted(tmp_dict.items(), key=lambda t: t[0]))
        for entry in sorted_tmp:
            predictions.append(sorted_tmp[entry][-1][0])
      #      print entry
      #      print sorted_tmp[entry][-1]
        return predictions

    def parse_for_names(self, predicted_words):
        #name = [e for e in predicted_words if any(x in e for x in self.word_that_are_names)]
        #print name
        #print 'car' in predicted_words
        for i, word in enumerate(predicted_words):
            if word in self.word_that_are_names:
                return i

if __name__ == '__main__':

    names_list = ['train', 'car', 'bus', 'truck', 'van', 'tram', 'boat']  #
    model = '/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_bus/'
    zs = Zero_Shooter(model, names_list)
    embeddings = helper.word_embeddings.Embeddings(model)
    word_model = embeddings.init_reduced_embeddings()

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    print len(zs.candidates)

    counter = 0
    for region_id in zs.candidates:
        region_id = str(region_id)
 #   region_id = '167959'
        sentence = zs.get_predictions(region_id)
        index =  zs.parse_for_names(sentence)
        if index == None:
            print sentence
            continue
        candidate_words_and_probs = zs.candidates[region_id][str(index + 1)]
        cand_words = [x[0] for x in candidate_words_and_probs]
        cand_probs = [float(x[1]) for x in candidate_words_and_probs]
        print cand_words
        print cand_probs
        new_vec = embeddings.words2embedding_weighted(cand_words, cand_probs)
        new_words_10 = embeddings.get_words_for_vector(new_vec, 10)
        print new_words_10
        new_words_5 = embeddings.get_words_for_vector(new_vec, 5)
        if "bus" in [x[0] for x in new_words_10]:
            hit_at_10 += 1
        if "bus" in [x[0] for x in new_words_5]:
            hit_at_5 += 1
        new_words_1 = embeddings.get_words_for_vector(new_vec, 1)
        if "bus" in [x[0] for x in new_words_1]:
            hit_at_1 += 1
        counter += 1

    print counter
    print "accuracy hit@1: ", hit_at_1 / float(counter)
    print "accuracy hit@5: ", hit_at_5 / float(counter)
    print "accuracy hit@10: ", hit_at_10 / float(counter)

