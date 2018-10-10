import numpy as np
from collections import defaultdict
import collections
import pandas as pd
import json
import params as p
import ast

# Script for preparing RefCOCO data for the training of the LSTM.

class Data:

    def __init__(self, res_dir, words_excluded= [], cats_excluded = [], additional_words = [], imgfeatpath = "../data/refcoco/mscoco_vgg19_refcoco.npz",
                 refspath = "../data/refcoco/refcoco_refdf.json.gz", splitpath = "../data/refcoco/refcoco_splits.json"):

        self.results_data_dir = res_dir
        self.imgfeatures_path = imgfeatpath
        self.refs_path = refspath
        self.split_path = splitpath
        self.discarded_words = []
        self.excluded_words = words_excluded
        self.excluded_categories_ids = cats_excluded
        self.additional = additional_words
        self.refs_moved_to_test = [] # ids of all those words ignored and put into test set

        self.load_data()
        self.prepare_data()
        self.clean_vocab()
        self.prepare_training()

        with open(self.results_data_dir + '/words_too_rare.json', 'w') as f:
            json.dump(self.discarded_words, f)
        with open(self.results_data_dir + '/refs_moved_to_test.json', 'w') as f:
            json.dump(self.refs_moved_to_test, f)

        if len(self.refs_moved_to_test) == 0:
            print "category not in training set"

    def load_data(self):
        ############ load and prepare image features ###########################

        # Image Features  --> numpy npz file includes one key/array, arr_0
        self.extracted_features = np.load(self.imgfeatures_path)['arr_0']

        ########### load and prepare referring expressions dataset ##############
        refcoco_data = pd.read_json(self.refs_path, orient="split", compression="gzip")
        with open(self.split_path) as f:
            splits = json.load(f)
        splitmap = {'val': 'val', 'train': 'train', 'testA': 'test', 'testB': 'test'}
        # for every group in split --> for every entry --> make entry in new dict
        # file2split just translates testA and testB to "test"?
        new_split_dict = {val: splitmap[key] for key in splits for val in splits[key]}

        # dict of objectids and ref exps
        self.obj2phrases = defaultdict(list)
        # dict of objectids and split (train,test or val)
        self.obj2split = {}
        split2obj = {'train': [], 'test': []}

        # iterate over json "entries"
        for index, row in refcoco_data.iterrows():
            # id is tuple of image and region id
            objectid = (row['image_id'], row['region_id'])
            self.obj2phrases[objectid].append(row['refexp'].split())
            self.obj2split[objectid] = new_split_dict[row['image_id']]

        # print "Objects",len(obj2phrases)

    ## match visual data with referring expressions
    #  and set up raw data with splits
    def prepare_data(self):
        img_counter = 0
        sentence_counter = 0
        selected_img_features = []
        test_list = []
        test_count = 0

        self.raw_dataset = {'train': {'filenames': list(), 'images': list(), 'captions': list()},
                      'val': {'filenames': list(), 'images': list(), 'captions': list()},
                      'test': {'filenames': list(), 'images': list(), 'captions': list()}, }
        for obj2phrases_item in self.obj2phrases:  # tqdm(obj2phrases):

            # [:,1] means: all indices of x along the first axis, but only index 1 along the second --> this list
            # comprehension filters out features for one image
            features_for_imageId = self.extracted_features[
                self.extracted_features[:, 1] == obj2phrases_item[0]]  # obj2phrases_item[0] is image id
            # this filters out features for the correct region
            features_for_objectId = features_for_imageId[
                features_for_imageId[:, 2] == obj2phrases_item[1]]  # obj2phrases_item[1] is region id

            if len(features_for_objectId) > 0:

                isIgnored = []
                image = np.array(features_for_objectId[0])[3:]  # due to data structure -> indices
                test_list.append(np.array(features_for_objectId[0])[3:])
                test_count += 1

                split = self.obj2split[obj2phrases_item]
                filename = "_".join([str(obj2phrases_item[0]), str(obj2phrases_item[1])])
                caption_group = []
                for ref in self.obj2phrases[obj2phrases_item]:
                    caption_group.append(ref)

                    for word in self.excluded_words:
                        if word in ref:
                            isIgnored = True

                    if obj2phrases_item[1] in self.excluded_categories_ids:
                        isIgnored = True

                image = image / np.linalg.norm(image)

                if not isIgnored:
                    self.raw_dataset[split]['filenames'].append(filename)
                    self.raw_dataset[split]['images'].append(image)
                    self.raw_dataset[split]['captions'].append(caption_group)
                    #for ref in caption_group:
                      #  if 'laptop' in ref:
                        #    print filename
                else:
                    self.raw_dataset['test']['filenames'].append(filename)
                    self.raw_dataset['test']['images'].append(image)
                    self.raw_dataset['test']['captions'].append(caption_group)
                    self.refs_moved_to_test.append(str(obj2phrases_item[1]))


        print 'raw data set', len(self.raw_dataset['train']['captions'])  # 42279
        print len(self.raw_dataset['val']['images'])
        print len(self.raw_dataset['test']['images'])


    # use only words with a minimum frequency for the vocabulary
    def clean_vocab(self):
        all_tokens = (token for caption_group in self.raw_dataset['train']['captions'] for caption in caption_group for token
                      in caption)

        token_freqs = collections.Counter(all_tokens)
        self.vocab = sorted(token_freqs.keys(), key=lambda token: (-token_freqs[token], token))

        with open(self.results_data_dir + '/token_freqs.json', 'w') as f:
            json.dump(token_freqs , f)

        print "all tokens count: ", len(self.vocab)
        # discard words with very low frequency
        while token_freqs[self.vocab[-1]] < p.min_token_freq:
            self.discarded_words.append(self.vocab.pop())

        self.vocab_size = len(self.vocab) + 2  # + edge and unknown tokens
        print('vocab:', self.vocab_size)

    # prepare captions and images in the correct format
    def parse(self, data):
        indexes = list()
        lens = list()
        images = list()
        for (caption_group, img) in zip(data['captions'], data['images']):
            for caption in caption_group:
                indexes_ = [self.token_to_index.get(token, self.unknown_index) for token in caption]
                indexes.append(indexes_)
                lens.append(len(indexes_) + 1)  # add 1 due to edge token
                images.append(img)

        maxlen = max(lens)

        in_mat = np.zeros((len(indexes), maxlen), np.int32)
        out_mat = np.zeros((len(indexes), maxlen), np.int32)
        for (row, indexes_) in enumerate(indexes):
            in_mat[row, :len(indexes_) + 1] = [self.edge_index] + indexes_
            out_mat[row, :len(indexes_) + 1] = indexes_ + [self.edge_index]
        return (in_mat, out_mat, np.array(lens, np.int32), np.array(images))

    # prepare data for the training: index and token allocation, sorting of images and captions for the splits
    def prepare_training(self):
        self.token_to_index = {token: i + 2 for (i, token) in enumerate(self.vocab)}
        self.index_to_token = {i + 2: token for (i, token) in enumerate(self.vocab)}

        self.index_to_token[1] = "UNKNOWN"
        self.edge_index = 0
        self.unknown_index = 1

        (self.train_captions_in, self.train_captions_out, self.train_captions_len, self.train_images) = self.parse(self.raw_dataset['train'])
        (self.val_captions_in, self.val_captions_out, self.val_captions_len, self.val_images) = self.parse(self.raw_dataset['val'])
        (self.test_captions_in, self.test_captions_out, self.test_captions_len, self.test_images) = self.parse(self.raw_dataset['test'])
        print("Train captions", np.shape(self.train_captions_in))



if __name__ == "__main__":
    data = Data()