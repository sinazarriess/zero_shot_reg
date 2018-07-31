import helper.word_embeddings as w
import helper.zero_shot as z
import csv
from collections import defaultdict
import json
import os
import ast

def generate_all_embeddingspaces(categories):
    for key in categories.keys():
        path = '/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_' + key + '/'
        if os.path.exists(path):
            embeddings_instance = w.Embeddings(path, True)
            if not os.path.exists(path + 'tmp_word2vec_only_names.txt'):
                print "Generating embeddings for", categories[key][0].strip()
                embeddings_instance.generate_reduced_w2v_file()

def do_zero_shot_all(categories):
    acc_mean_hit_at_1 = 0
    acc_mean_hit_at_2 = 0
    acc_mean_hit_at_5 = 0
    acc_mean_hit_at_10 = 0
    acc_mean_chance = 0
    acc_mean_before = 0

    words_not_in_glove = 0

    use_reduced_vector_space = True
    use_only_names = True
    numbr_candidates = 10
    print "Use only nouns in embedding space: ", use_only_names
    print "Number of vectors used for combination: ", numbr_candidates
    print "Reduced model: ", use_reduced_vector_space

    valid_categories = 0
    not_enough_samples = 0

    global_embeddings = w.Embeddings('', False)
    global_embeddings.get_global_model()

    for c in categories.keys():
        category_name = (categories[c][0]).strip()
        if len(global_embeddings.get_global_vector(category_name)) > 0:
            model = '/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_' + c + '/'
            if os.path.exists(model + 'inject_refcoco_refrnn_compositional_3_512_1/4eval_greedy.json'):
                test_regions = []
                with open(model + 'refs_moved_to_test.json', 'r') as f:
                    ids = f.readline()
                    test_regions = ast.literal_eval(ids)
                print "Number of test regions: ", len (test_regions)
                if len(test_regions) < 10:
                    print "not enough samples, only ", len (test_regions), ", skipping category ", category_name
                    not_enough_samples += 1
                    continue

                zs = z.Zero_Shooter(model, numbr_candidates)
                embed = w.Embeddings(model, use_only_names)
                if use_reduced_vector_space:
                    word_model = embed.init_reduced_embeddings()
                else:
                    word_model = embed.get_global_model()

                print "**** ",category_name , " ****"
                results =  zs.do_zero_shot(embed, category_name, use_reduced_vector_space)
                print "number of utterances to analyse: ", len(zs.candidates)
              #  print "valid sentences:", results[4], ',', round(results[4]/float(len(zs.candidates)) * 100, 2), '%'
                print "( Number of embeddings: ", len(word_model.vocab), ")"
                chance_acc = 1 / float(len(word_model.vocab))
                acc_mean_chance += chance_acc
                acc_mean_hit_at_1 += results[0]
                acc_mean_hit_at_2 += results[1]
                acc_mean_hit_at_5 += results[2]
                acc_mean_hit_at_10 += results[3]
                acc_mean_before += zs.bus_counter / float(results[4])

       #         print "chance: ",  round(chance_acc * 100, 4), '%'
       #         print "accuracy hit@1: ", round(results[0] * 100, 2) , '%'
       #         print "accuracy hit@2: ", round(results[1] * 100, 2) , '%'
                print "accuracy hit@5: ", round(results[2] * 100, 2) , '%'
       #         print "accuracy hit@10: ", round(results[3] * 100, 2) , '%'
       #         print "correct hits before: ", round(zs.bus_counter / float(results[4]) * 100, 2), '%\n'
                with open(model + 'zero_shot_refs_'+ str(c) + '.json', 'w') as f:
                    json.dump(zs.zero_shot_refs, f)

                valid_categories += 1
            else:
                print category_name, "is lacking"
        else:
            print "Category name not in GloVE embeddings (", category_name, ")"
            words_not_in_glove += 1

    if valid_categories > 0:
        print "Valid categories count: ", valid_categories
        valid_categories = float(valid_categories)
        mean_hit_at_1 = acc_mean_hit_at_1 / valid_categories
        print "Mean hit@1: ", acc_mean_hit_at_1, "/", valid_categories, " = ", round(mean_hit_at_1 * 100, 2), '%'
        mean_hit_at_2 = acc_mean_hit_at_2 / valid_categories
        print "Mean hit@2: ", acc_mean_hit_at_2, "/", valid_categories, " = ", round(mean_hit_at_2 * 100, 2), '%'
        mean_hit_at_5 =  acc_mean_hit_at_5 / valid_categories
        print "Mean hit@5: ", acc_mean_hit_at_5, "/", valid_categories, " = ", round(mean_hit_at_5 * 100, 2), '%'
        mean_hit_at_10 = acc_mean_hit_at_10 / valid_categories
        print "Mean hit@10: ", acc_mean_hit_at_10, "/", valid_categories, " = ", round(mean_hit_at_10 * 100, 2), '%'
        mean_before = acc_mean_before / valid_categories
        print "Mean correct before: ", round(mean_before * 100, 2), '%'

        print words_not_in_glove, " categories could not be trained because name not in GloVe (e.g. hair drier)"
        print not_enough_samples, " categories could not be used because there were not enough samples"


if __name__ == "__main__":
    categories = defaultdict()
    reader = csv.reader(open("../eval/cats.txt"))
    for row in reader:
        categories[row[0].strip()] = row[1:]

 #   categories['40'] = ['  baseball glove']
   # generate_all_embeddingspaces(categories)
    do_zero_shot_all(categories)