import numpy as np
import pandas as pd
import pickle
from scipy import stats
from tqdm import tqdm
from numpy.linalg import norm
import math
import torch 

import warnings
warnings.filterwarnings('ignore')

class DMM:

    def __init__(self, mmc_sim_file:str, word_index_file:str, embedding_matrix_file:str, concepts_file:str):
        """ The DMM sequence scoring method that we use in order to more efficiently rerank the generated sequences during beam searching.
        An illustration of the algorithm is provided in my Thesis paper.

        Args:
            hist_train_file (str): The name of the pickle file that contains the training maximum cosine similarity (between tags and relative captions) histogram.
            mmc_sim_file (str): The name of the pickle file that contains the median maximum cosine similarity value for each tag (based on calculations on the training dataset).
        """

        self.mmc_sim_file = mmc_sim_file

        self.mmc_sim = self.pickle_to_dict(self.mmc_sim_file)
        self.word_index = self.pickle_to_dict(word_index_file)
        self.embedding_matrix = np.load(embedding_matrix_file)
        self.respective_tags = list()
        self.centroid_embeddings, self.gen_tags_dict = dict(), dict()

        max_len = self.embedding_matrix.shape[0]
        self.word_index['startsequence'] = max_len + 1
        self.word_index['endsequence'] = max_len + 2
        self.word_index['<unk>'] = max_len + 3
        self.word_index['endofsequence'] = max_len + 4

        self.mmc_sim['Pneumopericardium'] = [0.85]

        for i in range(5):
            to_add = np.array([np.ones(self.embedding_matrix.shape[1])])
            self.embedding_matrix = np.append(self.embedding_matrix, to_add, axis=0)

        new_len = self.embedding_matrix.shape[0]

        concepts_mapper = pd.read_csv(concepts_file, sep="\t", header=None, names=['cui', 'concept'])

        # Build a mapper
        self._concepts_dict = {}
        for row in concepts_mapper['concept']:
            mapper = concepts_mapper.loc[concepts_mapper['concept'] == row].values.flatten().tolist()
            self._concepts_dict[mapper[0]] = mapper[1]
        


    def pickle_to_dict(self, file):

        # save dictionary to pickle file
        file_to_read = open(file, "rb")
        loaded_hist = pickle.load(file_to_read)

        return loaded_hist

    
    # Define function that calculates the given text's word embeddings centroid.
    def text_centroid(self, text, model, word_index):
        """ Calculate centroid function """
        text_vec =[]
        counter = 0
        text = text.split(" ")
        for word in text:
            try:
                if (counter == 0):
                    text_vec = model[word_index[word.lower()]]
                    counter+=1
                else:
                    text_vec = np.add(text_vec, model[word_index[word.lower()]])
                    counter+=1
            except:
                pass

        return np.asarray(text_vec) / counter

    
    # Define function that calculates the word embeddings of each item in the given list
    def get_concept_word_embeddings(self, _concepts:list, dims):

        concepts_embeddings = list()

        if dims == 2:

            concepts_embeddings = [[] for _ in _concepts]

            for c in clist:

                for char in '-.:[]()=/':
                    c = c.replace(char, ' ')

                if ((len(c.split(' ')) == 1)):
                    # if tag is only one word --> word_embedding(tag)
                    concepts_embeddings[i].append(
                        self.embedding_matrix[self.word_index[c.lower()]] if c.lower() in self.word_index else np.zeros(self.embedding_matrix.shape[1])
                    )
                else:
                    # else if tag is more than one word --> centroid of words embeddings of each tag subword
                    if c not in self.centroid_embeddings.keys():
                        centroid_emb = self.text_centroid(c, self.embedding_matrix, self.word_index)
                        concepts_embeddings[i].append(centroid_emb)
                        self.centroid_embeddings[c] = centroid_emb
                    else:
                        concepts_embeddings[i].append(self.centroid_embeddings[c])

        elif dims == 1:

            for i, c in enumerate(_concepts):

                key = c

                for char in ['-', '.', ':', '[', ']', '(', ')', '=', '/']:
                    c = c.replace(char, ' ')


                if ((len(c.split(' ')) == 1)):
                    if c.lower() in self.word_index:
                        concepts_embeddings.append(self.embedding_matrix[self.word_index[c.lower()]])
                        self.respective_tags.append(key)
                    else:
                        #not found in word index!
                        concepts_embeddings.append(np.zeros(self.embedding_matrix.shape[1]))
                else:
                    if c not in self.centroid_embeddings.keys():
                        centroid_emb = self.text_centroid(c, self.embedding_matrix, self.word_index)
                        concepts_embeddings.append(centroid_emb)
                        self.respective_tags.append(key)
                        self.centroid_embeddings[c] = centroid_emb
                    else:
                        concepts_embeddings.append(self.centroid_embeddings[c])
                        self.respective_tags.append(key)

        return concepts_embeddings


    
    def get_captions_word_embeddings(self, _captions:list, dims):

        captions_embeddings = list()

        if dims == 2:
            captions_embeddings = [[] for _ in _captions]
            for c in clist.split(' '):

                for char in '-.:[]()=/':
                    c = c.replace(char, ' ')



                if ((len(c.split(' ')) == 1)):
                    captions_embeddings[i].append(
                        self.embedding_matrix[self.word_index[c.lower()]] if c.lower() in self.word_index else np.zeros(self.embedding_matrix.shape[1])
                    )
                elif ((len(c.split()) > 1) and (len(self.text_centroid(c, self.embedding_matrix, self.word_index)) > 0)):
                    captions_embeddings[i].append(self.text_centroid(c, self.embedding_matrix, self.word_index))
                else:
                    captions_embeddings[i].append(np.zeros(self.embedding_matrix.shape[1]))

        elif dims == 1:
            for i, c in enumerate(_captions):

                for char in ['-', '.', ':', '[', ']', '(', ')', '=', '/']:
                    c = c.replace(char, ' ')

                caption_centroid = self.text_centroid(c, self.embedding_matrix, self.word_index)

                if ((len(c.split(' ')) == 1)):
                    captions_embeddings.append(
                        self.embedding_matrix[self.word_index[c.lower()]] if c.lower() in self.word_index else np.zeros(self.embedding_matrix.shape[1])
                    )
                elif ((len(c.split()) > 1) and (len(caption_centroid) > 0)):
                    captions_embeddings.append(caption_centroid)
                else:
                    captions_embeddings.append(np.zeros(self.embedding_matrix.shape[1]))


        return captions_embeddings


    def compute_sims(self, concepts_embeds:list, captions_embeds:list, concepts, flag=False):
        similarities = list()

        if (isinstance(concepts, str)):
            concepts = [concepts]

        for i, tags_i in enumerate(concepts):
            similarities.append([])
            for k in range(len(captions_embeds)):
                similarities[i].append(
                    self.cosine_sim(concepts_embeds if flag else concepts_embeds[i], captions_embeds[k])
                )

        return similarities

    def cosine_sim(self, A, B):

        # compute cosine similarity
        return np.dot(A,B)/(norm(A)*norm(B))

    
    def compute_hist(self, concepts_embeddings, captions_embeddings, gen_tags_dict, concepts):
        sims = self.compute_sims(concepts_embeddings, captions_embeddings, concepts, False)

        # iterate through the sims vector
        for k, rt in enumerate(sims):
            sims[k] = [x for x in sims[k] if (math.isnan(x))==False]
            if len(sims[k]) > 0:
                #print('respective tag:', self.respective_tags[k])
                if (concepts[k] in gen_tags_dict.keys()):
                    gen_tags_dict[concepts[k]].append(np.max(sims[k]))
                else:
                    #print('in else:')
                    gen_tags_dict[concepts[k]] = list()
                    gen_tags_dict[concepts[k]].append(np.max(sims[k]))
            else:
                print('Empty sims list!!!')

        return gen_tags_dict


    def compute_histogram_divergence(self, train_hist, gen_hist):

        score = 0
        #print('tags for hist:', gen_hist.keys())
        for key in gen_hist.keys():
            train_list = list(train_hist[key])
            gen_list = list(gen_hist[key])

            #KS-test looks suitable!
            ks = stats.kstest(train_list, gen_list)
            score += ks[0]

        aggregated_score = score / len(gen_hist.keys())
        return aggregated_score



    def dmm_loss(self, caption_embeds, concept_embeds, concept):

        cos_t = self.compute_sims(concept_embeds, caption_embeds, concept, True)

        cos_t[0] = [x for x in cos_t[0] if str(x) != 'nan']

        max_cos_t = np.max(cos_t[0])

        max_cos_c = self.mmc_sim.get(concept, [0.5])[0]

        dmm = (max_cos_t - max_cos_c) ** 2
        return dmm

    def check_for_nan(self, t):

        return torch.is_tensor(t) or t == ''


    def dmm_handler(self, caption, concepts):
        """For a given caption and each assigned concepts, calculate the dmm loss, as defined in my MSc Thesis.
        
        Args:
            caption: The caption (in text format) for which we want to calculate the dmm loss.
            concepts: A list of the concepts (in text format) that are assigned to the given caption.
        """

        concepts = [] if self.check_for_nan(concepts[0]) else concepts[0].split(';')

        concepts = [self._concepts_dict[c] for c in concepts]

        caption_embeddings = self.get_captions_word_embeddings(caption, dims=1)
        concept_embeddings = self.get_concept_word_embeddings(concepts, dims=1)

        dmm_loss_sum = 0
        for i, c in enumerate(concepts):
            dmm_loss_sum += self.dmm_loss(caption_embeddings, concept_embeddings[i], concepts[i])

        return dmm_loss_sum





    
