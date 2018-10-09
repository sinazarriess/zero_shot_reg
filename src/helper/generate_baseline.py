import json
import csv
from collections import defaultdict, OrderedDict
import os
from pprint import pprint


class Baseline:

    def __init__(self):
        self.modelspath = '/mnt/Data/zero_shot_reg/src/eval/new_models/with_reduced_cats_'
      #  self.modelspath = '/mnt/Data/zero_shot_reg/src/eval/model/with_reduced_cats_'

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
    reader = csv.reader(open("../eval/cats.txt"))
    for row in reader:
        categories[row[0].strip()] = row[1:]

    for c in categories:
        c = str(c)
        results[c] = defaultdict()
        results[c]["using top 1"] = b.generate('1', c)
        results[c]["using top 5"] = b.generate('5', c)
        results[c]["using top 10"] = b.generate('10', c)

    pprint(results)
    with open('/mnt/Data/zero_shot_reg/src/eval/new_models/baseline.json', 'w') as f:
        json.dump(results, f)



