import json
import csv
from collections import defaultdict, OrderedDict
import os
from pprint import pprint


## Script to evaluate the "raw" baseline data generated by the zero-shot script.
## The purpose is to find the 5 most frequent predictions for a single category, which can be used
## for a comparison with another model (WAC).

baseline_path = '/mnt/Data/zero_shot_reg/src/eval/new_models/baseline.json'
modelspath = '/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_'
cats_path = "../eval/cats.txt"

class Baseline:

    def __init__(self):
        self.modelspath = modelspath

    def generate(self, topN, c):
        self.tempdict = defaultdict()
        path = self.modelspath + c + '/' + 'baseline_frequencies_top' + topN + '.json'
        if os.path.exists(path):
            with open(path) as f:
                freqs = json.load(f)
                number = sum(freqs.values())
                top_words = OrderedDict(sorted(freqs.items(), key=lambda t: t[1])).items()[-5:]

                self.tempdict["top words"] = top_words
                self.tempdict["total number"] = number
                return self.tempdict
        else:
            print "path does not exist: ", path
            return


if __name__ == "__main__":
    b = Baseline()
    results = defaultdict()

    categories = defaultdict()
    reader = csv.reader(open(cats_path))
    for row in reader:
        categories[row[0].strip()] = row[1:]

    for c in categories:
        c = str(c)
        results[c] = defaultdict()
        results[c]["using top 1"] = b.generate('1', c)
        results[c]["using top 5"] = b.generate('5', c)
        results[c]["using top 10"] = b.generate('10', c)

    pprint(results)
    with open(baseline_path, 'w') as f:
        json.dump(results, f)



