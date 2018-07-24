import helper.word_embeddings as w
import helper.zero_shot as z
import csv
from collections import defaultdict
import json


def generate_all_embeddingspaces(categories):
    for key in categories.keys():
        path = '/mnt/Data/zero_shot_reg/src/eval/new_models/without_' + key
        embeddings_instance = w.Embeddings(path)
        embeddings_instance.generate_reduced_w2v_file()

def do_zero_shot_all(categories):

    acc_mean_hit_at_1 = 0
    acc_mean_hit_at_2 = 0
    acc_mean_hit_at_5 = 0
    acc_mean_hit_at_10 = 0
    acc_mean_chance = 0

    use_reduced_vector_space = True
    use_only_names = True
    numbr_candidates = 10
    print "Number of vectors used for combination: ", numbr_candidates
    print "Reduced model: ", use_reduced_vector_space

    for c in categories.keys():
        model = '/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_' + categories[c] + '/'
        zs = z.Zero_Shooter(model, numbr_candidates)
        embed = w.Embeddings(model, use_only_names)
        if use_reduced_vector_space:
            word_model = embed.init_reduced_embeddings()
        else:
            word_model = embed.get_global_model()

       # print zs.words_that_are_names
        print "**** ", c
        results =  zs.do_zero_shot(embed, c, use_reduced_vector_space)
        print "number of utterances to analyse: ", len(zs.candidates)
      #  print "valid sentences:", results[4], ',', round(results[4]/float(len(zs.candidates)) * 100, 2), '%'
        print "( Number of embeddings: ", len(word_model.vocab), ")"
        chance_acc = 1 / float(len(word_model.vocab))
        acc_mean_chance += chance_acc
        acc_mean_hit_at_1 += results[0]
        acc_mean_hit_at_2 += results[1]
        acc_mean_hit_at_5 += results[2]
        acc_mean_hit_at_10 += results[3]

        print "chance: ",  round(chance_acc * 100, 4), '%'
        print "accuracy hit@1: ", round(results[0] * 100, 2) , '%'
        print "accuracy hit@2: ", round(results[1] * 100, 2) , '%'
        print "accuracy hit@5: ", round(results[2] * 100, 2) , '%'
        print "accuracy hit@10: ", round(results[3] * 100, 2) , '%'
        print "correct hits before: ", round(zs.bus_counter / float(results[4]) * 100, 2), '%\n' #TODO counter changed

        with open(model + 'zero_shot_refs_'+ str(c) + '.json', 'w') as f:
            json.dump(zs.zero_shot_refs, f)

    nmbr_categories = float(len(categories.keys()))
    mean_hit_at_1 = acc_mean_hit_at_1 / nmbr_categories
    print "Mean hit@1: ", acc_mean_hit_at_1, "/", nmbr_categories, " = ", round(mean_hit_at_1 * 100, 2), '%'
    mean_hit_at_2 = acc_mean_hit_at_2 / nmbr_categories
    print "Mean hit@1: ", acc_mean_hit_at_2, "/", nmbr_categories, " = ", round(mean_hit_at_2 * 100, 2), '%'
    mean_hit_at_5 =  acc_mean_hit_at_5 / nmbr_categories
    print "Mean hit@1: ", acc_mean_hit_at_5, "/", nmbr_categories, " = ", round(mean_hit_at_5 * 100, 2), '%'
    mean_hit_at_10 = acc_mean_hit_at_10 / nmbr_categories
    print "Mean hit@1: ", acc_mean_hit_at_10, "/", nmbr_categories, " = ", round(mean_hit_at_10 * 100, 2), '%'


if __name__ == "__main__":
    categories = defaultdict()
    reader = csv.reader(open("cats.txt"))
    for row in reader:
        categories[row[0].strip()] = row[1:]

    generate_all_embeddingspaces(categories)

    do_zero_shot_all(categories)