import json
import helper.word_embeddings
from collections import OrderedDict, defaultdict
from nltk import word_tokenize, UnigramTagger
from nltk.corpus import brown

class Zero_Shooter:

    def __init__(self, modelpath, candidates):
        self.bus_counter = 0
        with open(modelpath + 'all_highest_probs_'+ str(candidates) + '.json', 'r') as f:
            self.candidates = json.load(f)
        with open(modelpath + 'restoredmodel_refs_greedy.json', 'r') as f:
            self.refs = json.load(f)
        self.word_that_are_names = list()
        with open("./noun_list_long.txt", 'r') as f:
            for row in f.readlines():
               self.word_that_are_names.append(row.strip())
        self.unigram_tagger = UnigramTagger(brown.tagged_sents())
        self.zero_shot_refs = defaultdict()

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
            # or return this ^-
            if word in self.word_that_are_names:  ## always returns first instance ...
                return i
        #print predicted_words
        return -1

    def parse_pos(self, tokens, cat):
       # tokens = word_tokenize(sentence)
        tags = self.unigram_tagger.tag(tokens)
        nouns = [x for x in tags if x[1] == 'NN']
        if len(nouns) > 1:
            print tokens
        if len(nouns) > 0:
            if nouns[0][0] == cat:
                self.bus_counter += 1
            return tokens.index(nouns[0][0])  # to keep it easy - if two nouns, this is a simplification!
        else:
            unknown_nouns = [x for x in tags if x[1] == 'None']
            if len(unknown_nouns) > 0:
                return tokens.index(unknown_nouns[0])
            else:
                return -1

    def do_zero_shot(self, embeddings, category):
        hit_at_1 = 0
        hit_at_2 = 0
        hit_at_5 = 0
        hit_at_10 = 0

        counter = 0
        for region_id in self.candidates:
            region_id = str(region_id)
            #   region_id = '167959'
            sentence = self.get_predictions(region_id)

            ## use pos tagger
            #index = self.parse_pos(sentence, category)

            ## OR use name list
            index = self.parse_for_names(sentence, category)

            if index < 0:
           #     print sentence
                self.zero_shot_refs[region_id] = self.refs[region_id]
                continue

            candidate_words_and_probs = self.candidates[region_id][str(index + 1)]
            cand_words = [x[0] for x in candidate_words_and_probs]
            cand_probs = [float(x[1]) for x in candidate_words_and_probs]

            new_vec = embeddings.words2embedding_weighted(cand_words, cand_probs)
            new_words_10 = embeddings.get_words_for_vector(new_vec, 10)
            new_words_5 = embeddings.get_words_for_vector(new_vec, 5)
            if category in [x[0] for x in new_words_10]:
                hit_at_10 += 1
            if category in [x[0] for x in new_words_5]:
                hit_at_5 += 1
            new_words_1 = embeddings.get_words_for_vector(new_vec, 1)
            if category in [x[0] for x in new_words_1]:
                hit_at_1 += 1
            new_words_2 = embeddings.get_words_for_vector(new_vec, 2)
            if category in [x[0] for x in new_words_2]:
                hit_at_2 += 1
            counter += 1

            ref = self.refs[region_id][0].split()
            ref[index] = new_words_1[0][0]
            new_ref = ' '.join(ref)
            self.zero_shot_refs[region_id] = [new_ref]

        return hit_at_1/ float(counter), hit_at_2/ float(counter), hit_at_5/ float(counter), hit_at_10/ float(counter), counter


if __name__ == '__main__':

    cats = ['laptop', 'bus', 'horse']

    use_reduced_vector_space = True
    numbr_candidates = 10
    print "Number of vectors used for combination: ", numbr_candidates
    print "Reduced model: ", use_reduced_vector_space

    for c in cats:
        model = '/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_' + c + '/'
        zs = Zero_Shooter(model, numbr_candidates)
        embed = helper.word_embeddings.Embeddings(model)
        if use_reduced_vector_space:
            word_model = embed.init_reduced_embeddings()
        else:
            word_model = embed.get_global_model()

        print "**** ", c
        results =  zs.do_zero_shot(embed, c)
        print "number of utterances to analyse: ", len(zs.candidates)
        print "valid sentences:", results[4], ',', round(results[4]/float(len(zs.candidates)) * 100, 2), '%'
        print "( Number of embeddings: ", len(word_model.vocab), ")"
        chance_acc = 1 / float(len(word_model.vocab))
        print "chance: ",  round(chance_acc * 100, 4), '%'
        print "accuracy hit@1: ", round(results[0] * 100, 2) , '%'
        print "accuracy hit@2: ", round(results[1] * 100, 2) , '%'
        print "accuracy hit@5: ", round(results[2] * 100, 2) , '%'
        print "accuracy hit@10: ", round(results[3] * 100, 2) , '%'
        print "correct hits before: ", round(zs.bus_counter / float(results[4]) * 100, 2), '%\n'

        with open(model + 'zero_shot_refs_'+ str(c) + '.json', 'w') as f:
            json.dump(zs.zero_shot_refs, f)







