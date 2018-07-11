import json
import helper.word_embeddings
from collections import OrderedDict
from nltk import pos_tag

class Zero_Shooter:

    def __init__(self, modelpath, candidates):
        self.bus_counter = 0
        with open(modelpath + 'all_highest_probs_'+ str(candidates) + '.json', 'r') as f:
            self.candidates = json.load(f)
        self.word_that_are_names = list()
        with open("./noun_list_long.txt", 'r') as f:
            for row in f.readlines():
               self.word_that_are_names.append(row.strip())

    def get_predictions(self, region_id):
        predictions = list()
        tmp_dict = self.candidates[region_id]
        sorted_tmp = OrderedDict(sorted(tmp_dict.items(), key=lambda t: t[0]))
        for entry in sorted_tmp:
            predictions.append(sorted_tmp[entry][-1][0])
      #      print entry
      #      print sorted_tmp[entry][-1]
        return predictions

    def parse_for_names(self, predicted_words, cat):   #TODO "man on horse"  ... POS tagging?
        for i, word in enumerate(predicted_words):
            if word == cat:
                self.bus_counter += 1
            if word in self.word_that_are_names:  ## always returns first instance ... 
                return i

    def do_zero_shot(self, embeddings, category):
        hit_at_1 = 0
        hit_at_2 = 0
        hit_at_5 = 0
        hit_at_10 = 0

        print "number of utterances to analyse: ", len(self.candidates)
        counter = 0
        for region_id in self.candidates:
            region_id = str(region_id)
            #   region_id = '167959'
            sentence = self.get_predictions(region_id)
            index = self.parse_for_names(sentence, category)
            if index == None:
           #     print sentence
                continue
            candidate_words_and_probs = self.candidates[region_id][str(index + 1)]
            cand_words = [x[0] for x in candidate_words_and_probs]
            cand_probs = [float(x[1]) for x in candidate_words_and_probs]
         #   print cand_words
         #   print cand_probs
            new_vec = embeddings.words2embedding_weighted(cand_words, cand_probs)
            new_words_10 = embeddings.get_words_for_vector(new_vec, 10)
         #   print new_words_10
            new_words_5 = embeddings.get_words_for_vector(new_vec, 5)
            if category in [x[0] for x in new_words_10]:
                hit_at_10 += 1
            if category in [x[0] for x in new_words_5]:
                hit_at_5 += 1
            new_words_1 = embeddings.get_words_for_vector(new_vec, 1)
            if category in [x[0] for x in new_words_1]:
                hit_at_1 += 1
            new_words_1 = embeddings.get_words_for_vector(new_vec, 2)
            if category in [x[0] for x in new_words_1]:
                hit_at_2 += 1
            counter += 1

        return hit_at_1/ float(counter), hit_at_2/ float(counter), hit_at_5/ float(counter), hit_at_10/ float(counter), counter


if __name__ == '__main__':

    cats = ['laptop', 'bus', 'horse']

    numbr_candidates = 100

    for c in cats:
        model = '/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_' + c + '/'
        zs = Zero_Shooter(model, numbr_candidates)
        embed = helper.word_embeddings.Embeddings(model)
        word_model = embed.init_reduced_embeddings()
        #word_model = embed.get_global_model()

        print "**** ", c
        results =  zs.do_zero_shot(embed, c)
        print "valid sentences:", results[4]
        print "accuracy hit@1: ", round(results[0] * 100, 2) , '%'
        print "accuracy hit@2: ", round(results[1] * 100, 2) , '%'
        print "accuracy hit@5: ", round(results[2] * 100, 2) , '%'
        print "accuracy hit@10: ", round(results[3] * 100, 2) , '%'
        print "correct hits before: ", round(zs.bus_counter / float(results[4]) * 100, 2), '%\n'







