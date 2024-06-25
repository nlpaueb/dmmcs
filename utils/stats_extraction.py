import pandas as pd
import numpy as np
import math
import os
from numpy.linalg import norm
import pickle

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import math
from tqdm import tqdm

import json, argparse, statistics

# STEP 1: Run python -m spacy download en_core_web_sm

class DMMCSStatsExtractor():
    def __init__(self, config_path):
        self.config_path = config_path

        # Load configurations from a JSON file
        self.load_config(config_path)

    
    def load_config(self, config_path):
        """ Load all arguments from a JSON configuration file """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    # Define function that tokenizes the given structure using the SpaCy library
    def tokenize_samples(self, samples, nlp):

        tokenized_samples = []
        for i in range(len(samples)):
            doc = nlp(samples[i])  # Tokenize the sample into sentences
            tokens = []
            for sent in doc.sents:
                for tok in sent:  # Iterate through the words of the sentence
                    if '\n' in tok.text or "\t" in tok.text or "--" in tok.text or "*" in tok.text or tok.text.lower() in STOP_WORDS:
                        continue
                    if tok.text.strip():
                        tokens.append(tok.text.replace('"',"'").strip())
            tokenized_samples.append(tokens)

        return tokenized_samples

    # Define function that calculates the given text's word embeddings centroid.
    def text_centroid(self, text, model, word_index):
        """ Calculate centroid function """
        text_vec =[]
        counter = 0
        text = text.split(" ")
        for word in text:
            try:
                if (counter == 0):
                    text_vec = model[self.word_index[word.lower()]]
                else:
                    text_vec = np.add(text_vec, model[self.word_index[word.lower()]])
                counter+=1
            except:
                pass

        return np.asarray(text_vec) / counter

    # Define function that calculates the word embeddings of each item in the given list
    def get_concept_word_embeddings(self, _concepts:list, dims):

        concepts_embeddings = list()
        if dims == 2:
            for i, clist in enumerate(_concepts):
                concepts_embeddings.append([])
                for c in clist:
                    c = c.replace('-', ' ')
                    c = c.replace('.', ' ')
                    c = c.replace(':', ' ')
                    c = c.replace('[', ' ')
                    c = c.replace(']', ' ')
                    c = c.replace('(', ' ')
                    c = c.replace(')', ' ')
                    c = c.replace('=', ' ')
                    c = c.replace('/', ' ')


                    if ((len(c.split(' ')) == 1)):
                        # if tag is only one word --> word_embedding(tag)
                        if c.lower() in self.word_index:
                            print(self.embedding_matrix[self.word_index[c.lower()]])
                            concepts_embeddings[i].append(self.embedding_matrix[self.word_index[c.lower()]])
                        else:
                            concepts_embeddings[i].append(np.ones(200))
                    else:
                        # else if tag is more than one word --> centroid of words embeddings of each tag subword
                        concepts_embeddings[i].append(self.text_centroid(c, self.embedding_matrix, self.word_index))
        elif dims == 1:

            for i, c in enumerate(_concepts):

                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')

                if ((len(c.split(' ')) == 1)):
                    # if tag is only one word --> word_embedding(tag)
                    if c.lower() in self.word_index:
                        #print(embedding_matrix[word_index[c.lower()]])
                        concepts_embeddings.append(self.embedding_matrix[self.word_index[c.lower()]])
                    else:
                        concepts_embeddings.append(np.zeros(200))
                else:
                    # else if tag is more than one word --> centroid of words embeddings of each tag subword
                    concepts_embeddings.append(self.text_centroid(c, self.embedding_matrix, self.word_index))

        return concepts_embeddings

    
    def get_captions_word_embeddings(self, _captions:list, dims):

        captions_embeddings = list()

        if dims == 2:
            for i, clist in enumerate(_captions):
                captions_embeddings.append([])
                for c in clist.split(' '):
                    c = c.replace('-', ' ')
                    c = c.replace('.', ' ')
                    c = c.replace(':', ' ')
                    c = c.replace('[', ' ')
                    c = c.replace(']', ' ')
                    c = c.replace('(', ' ')
                    c = c.replace(')', ' ')
                    c = c.replace('=', ' ')
                    c = c.replace('/', ' ')


                    if ((len(c.split(' ')) == 1)):
                        if c.lower() in self.word_index:
                            captions_embeddings[i].append(self.embedding_matrix[self.word_index[c.lower()]])
                        else:
                            captions_embeddings[i].append(np.ones(200))
                    elif ((len(c.split()) > 1) and (len(self.text_centroid(c, self.embedding_matrix, self.word_index)) > 0)):
                        captions_embeddings[i].append(self.text_centroid(c, self.embedding_matrix, self.word_index))
                    else:
                        captions_embeddings[i].append(np.ones(200))

        elif dims == 1:
            for i, c in enumerate(_captions):
                c = c.replace('-', ' ')
                c = c.replace('.', ' ')
                c = c.replace(':', ' ')
                c = c.replace('[', ' ')
                c = c.replace(']', ' ')
                c = c.replace('(', ' ')
                c = c.replace(')', ' ')
                c = c.replace('=', ' ')
                c = c.replace('/', ' ')

                if ((len(c.split(' ')) == 1)):
                    captions_embeddings.append(self.embedding_matrix[self.word_index[c.lower()]])
                elif ((len(c.split()) > 1) and (len(self.text_centroid(c, self.embedding_matrix, self.word_index)) > 0)):
                    captions_embeddings.append(self.text_centroid(c, self.embedding_matrix, self.word_index))
                else:
                    captions_embeddings[i].append(np.ones(200))


        return captions_embeddings

    
    # compute cosine similarity
    def cosine_sim(self, A, B):
        cosine = np.dot(A,B)/(norm(A)*norm(B))
        return cosine

    
    # Function that computes the cosine similarity betwen each tag and each caption word.
    def compute_sims(self, concepts_embeds:list, captions_embeds:list):

        similarities = list()
        for i, tags_i in enumerate(concepts_embeds):
            similarities.append([])
            for k in range(len(captions_embeds)):
                similarities[i].append(self.cosine_sim(concepts_embeds[i], captions_embeds[k]))
        return similarities

    def run(self):

        df=pd.read_csv(self.config["all_data"], sep='\t', encoding='latin')
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

        concepts_mapper = pd.read_csv(self.config["dataset_concepts_mapper"], sep="\t", header=None, names=['cui', 'concept'])

        # Build a mapper
        _concepts_dict = {}
        for row in concepts_mapper['concept']:
            mapper = concepts_mapper.loc[concepts_mapper['concept'] == row].values.flatten().tolist()
            _concepts_dict[mapper[0]] = mapper[1]

        # Create new column and fill it with nan values
        df['concepts'] = np.nan

        # Iterate through the dataframe and fill the column with either the real-world medical concept (if available), or the CUI.
        for i, cuis in enumerate(df['cuis']):
            tags = []
            for tag in cuis.split(';'):
                if tag in _concepts_dict.keys():
                    tags.append(_concepts_dict[tag])
                else:
                    tags.append(tag)
            df['concepts'][i] = ';'.join(tags)


        
        tags_dict = dict()
        for concepts in df['concepts']:
            tags = concepts.split(';')
            for t in tags:
                if t not in tags_dict.keys():
                    tags_dict[t] = 0


        # Load fasttext embeddings
        fasttext_embed = np.load(self.config["fasttext_embeddings"])
        fasttext_word_to_index = pickle.load(open(self.config["fasttext_vocabulary"], 'rb'))



        # save concepts and captions into list
        df_concepts, df_captions = list(df['concepts']), list(df['caption'])
        for i, item in enumerate(df_concepts):
            df_concepts[i] = item.split(';')



        nlp = spacy.load('en_core_web_sm',disable=["tagger", "parser","ner"])
        nlp.add_pipe('sentencizer')

        # Tokenize the captions
        df_captions_tokenized = self.tokenize_samples(df_captions, nlp)



        MAX_WORDS = 50000
        MAX_SEQUENCE_LENGTH = 150
        EMBEDDING_DIM = fasttext_embed.shape[1]

        # Init tokenizer
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='__UNK__')
        # num_words: the maximum number of words to keep, based on word frequency.
        # oov_token: will be used to replace OOV WORDS

        # Fit tokenizer (Updates internal vocabulary based on a list of texts.)
        tokenizer.fit_on_texts([" ".join(x) for x in df_captions_tokenized])

        # Converts text to sequences of IDs
        train_seqs = tokenizer.texts_to_sequences([" ".join(x) for x in df_captions_tokenized])

        # Pad the training sequences
        train_data = pad_sequences(train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


        #--------------------------------------------------------------------------------------------

        # Init tokenizer
        tokenizer2 = Tokenizer(num_words=len(df_concepts), oov_token='__UNK__')
        # num_words: the maximum number of words to keep, based on word frequency.
        # oov_token: will be used to replace OOV WORDS

        # Fit tokenizer (Updates internal vocabulary based on a list of texts.)
        tokenizer2.fit_on_texts([" ".join(x) for x in df_concepts])

        # Converts text to sequences of IDs
        train_seqs2 = tokenizer2.texts_to_sequences([" ".join(x) for x in df_concepts])

        # Pad the training sequences
        train_data2 = pad_sequences(train_seqs2, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


        # Save the word index from TensorFlow's tokenizer
        self.word_index = tokenizer.word_index
        self.word_index2 = tokenizer2.word_index

        print('Found {} unique tokens.\n'.format(len(self.word_index)))
        print('Found {} unique tokens2.\n'.format(len(self.word_index2)))

        self.word_index.update(self.word_index2)
        print('Found {} unique tokens2.\n'.format(len(self.word_index)))

        with open('../snapshots/artifacts/word_index.pkl', 'wb') as file:
            pickle.dump(self.word_index, file, protocol=pickle.HIGHEST_PROTOCOL)



        print("EMBEDDING DIM:", EMBEDDING_DIM)
        self.embedding_matrix = np.zeros((len(self.word_index)+2, EMBEDDING_DIM))  # +2 (pad, unkown)

        for word, i in self.word_index.items():
            if i > len(self.word_index):
                    continue
            try:
                embedding_vector = fasttext_embed[fasttext_word_to_index[word],:]
                self.embedding_matrix[i] = embedding_vector
            except:
                #pass
                #print("The embedding matrix for this word remains zero-full.")
                self.embedding_matrix[i] = np.ones(200)
        print('Size of Embedding Matrix:', len(self.embedding_matrix))
        print('Embedding Matrix:', self.embedding_matrix)

        np.save('../snapshots/artifacts/embedding_matrix.npy', self.embedding_matrix)



        # Start iterating through all the captions.
        respective_tags = list()
        concepts_embeddings = list()
        captions_embeddings = self.get_captions_word_embeddings(df_captions, dims=2)
        for i, caption in enumerate(df_captions): #df_captions is a list of length 71355, where each list item is one caption.
            tags = df_concepts[i]
            tags_embeddings = self.get_concept_word_embeddings(tags, dims=1)
            concepts_embeddings.append(tags_embeddings)
            respective_tags.append(tags)


        tags_dict = dict()
        for concepts in df['concepts']:
            tags = concepts.split(';')
            for t in tags:
                if t not in tags_dict.keys():
                    tags_dict[t] = list()


        
        # iterate through the dataset captions
        for i in tqdm(range(len(captions_embeddings))):
            # for each caption compute the cosine similarity between each tag and each caption word
            # ie. if #tags = 2 and len(caption)=10, then a matrix of size (2, 10) is returned
            sims = self.compute_sims(concepts_embeddings[i], captions_embeddings[i])

            # iterate through the sims vector
            for k, rt in enumerate(sims):
                sims[k] = [x for x in sims[k] if (math.isnan(x))==False]
                if len(sims[k]) > 0:
                    tags_dict[respective_tags[i][k]].append(np.max(sims[k]))
                else:
                    print('Empty sims list!')


        
        with open('../snapshots/artifacts/hist_train.pkl', 'wb') as file:
            pickle.dump(tags_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



        tags_dict2 = dict()
        for k in tags_dict.keys():
            counter = 0
            if len(tags_dict[k]) > 0:
                tags_dict2[k] = [statistics.median(tags_dict[k]), statistics.stdev(tags_dict[k]), len(tags_dict[k])]
            else:
                counter += 1


        sorted_dict2 = {k: v for k, v in sorted(tags_dict2.items(), key=lambda item: item[1][0], reverse=False)}

        # save dictionary to pickle file
        with open('../snapshots/artifacts/median_max_cos_c.pkl', 'wb') as file:
            pickle.dump(sorted_dict2, file, protocol=pickle.HIGHEST_PROTOCOL)








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to run InstructBLIP with JSON config.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    stats_extractor = DMMCSStatsExtractor(args.config)
    
    stats_extractor.run()

    