__author__ = 'EricLiu'

import shelve
import re
from json import load
from math import log
from operator import itemgetter
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer


class DatabaseBuilder:
    """
    Build the search database from json file
    provide search interface for boolean search
    using tf-idf to compute score
    """
    def __init__(self, raw_file):
        self.doc_dics = self.get_info_from_file(raw_file)
        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')

        # for computing document frequency
        self.doc_term_dic = {}

        # for store or read data from file
        self.bs_data = shelve.open('BooleanSearch.db')
        self.doc_data = {}
        self.term_postings = {}

    def build_database(self, from_file=True):
        """
        build the database in memory for searching
        :param from_file: True: read from shelve stored database built before
        :return:
        """
        if from_file:
            print 'Reading database from file'
            self.doc_data = self.bs_data['doc_data']
            self.term_postings = self.bs_data['term_postings']
            self.bs_data.close()
        else:
            print 'Start building database'
            doc_id = 0
            doc_length = len(self.doc_dics)
            for doc_dic in self.doc_dics.itervalues():
                if doc_id % 10 == 0:
                    print '%0.2f%%' % (100.0 * doc_id/doc_length)
                doc_id += 1
                self.build_doc_dic(doc_id, doc_dic)
                # process the document and add to term-posting list
                doc_detail = doc_dic['title']+' '+doc_dic['text']
                processed_doc = self.pre_process(doc_detail)
                self.build_term_dic(doc_id, processed_doc)
                # store for compute df
                self.doc_term_dic[doc_id] = processed_doc
            self.compute_df()
            self.bs_data['doc_data'] = self.doc_data
            self.bs_data['term_postings'] = self.term_postings
            self.bs_data.close()

    def pre_process(self, doc):
        """
        pre process the single document text
        :param doc: a document that constructed by title + text
        :return: a list of terms that contains parsed doc
        """
        doc = doc.lower()
        tokens = self.tokenizer.tokenize(doc)
        stop_removed = [word for word in tokens if word not in self.stop_words]
        stemmed = [self.stemmer.stem(word) for word in stop_removed]
        return stemmed

    def build_doc_dic(self, doc_id, doc_dic):
        self.doc_data[doc_id] = doc_dic

    def build_term_dic(self, doc_id, terms):
        """
        compute term frequency and store them to term-postings list
        for each term appears in a doc
        :param doc_id:
        :param terms:
        :return:
        """
        # for computing term frequency
        counter = Counter(terms)

        # store postings and tf for each term
        length = len(terms)
        for term, freq in counter.iteritems():
            tf = 1 + 1.0 * freq / length
            if term in self.term_postings.iterkeys():
                term_obj = self.term_postings[term]
            else:
                term_obj = Term()
            term_obj.add_posting(doc_id, tf)
            self.term_postings[term] = term_obj

    def compute_df(self):
        """
        compute the document frequency for each term
        :return: None
        """
        doc_length = len(self.doc_dics)
        for term in self.term_postings.itervalues():
            df = log(1 + 1.0 * len(term.get_postings()) / doc_length)
            term.set_df(df)

    @staticmethod
    def get_info_from_file(file_name):
        """load json from a file to python objective"""
        with open(file_name) as outfile:
            docs = load(outfile)
            return docs


class Term:
    """
    A term class contains a term's attributes: it's df score and a posting list
    the posting list is a dictionary, whose key is the document id and value is
    correspond term frequency score
    This should be the value of term_postings dictionary whose key is the term itself
    """
    def __init__(self):
        self.df = 0
        self.postings = {}

    def set_df(self, df):
        self.df = df

    def add_posting(self, doc_id, tf):
        self.postings[doc_id] = tf

    def get_postings(self):
        return self.postings.copy()

    def get_df(self):
        return self.df


class SearchEngine:
    def __init__(self, database=None):
        """
        If database is not provided, it will create a database from file 'wiki_all.txt'
        :param database: a DatabaseBuilder Class Object
        :return:
        """
        if database is None:
            database = DatabaseBuilder('wiki_all.txt')
            database.build_database(from_file=True)

        self.term_postings = database.term_postings
        self.pre_process = database.pre_process
        self.doc_data = database.doc_data

    def search(self, query_string):
        """
        find the intersection of input term's postings, get idf
        from the postings, find the word's tf and snippets
        compute the accumulated score, and display the result
        :param query_string: query string
        :return: None
        """
        query_list = self.pre_process(query_string)
        query_list = [term for term in query_list if term in self.term_postings.iterkeys()]
        # get the first term's postings as the first posting dic
        query_len = len(query_list)
        if query_len == 0:
            print 'No result found'
            return
        postings_dic = self.term_postings[query_list[0]].get_postings()
        if query_len > 1:
            for i in range(1, len(query_list)):
                postings_dic = self.dic_intersect(postings_dic, self.term_postings[query_list[i]].get_postings())

        doc_score_snippet = []
        for doc_id in postings_dic.iterkeys():
            score = self.get_score(doc_id, query_list)
            snippet = self.get_snippet(doc_id, query_list)
            doc_score_snippet.append((doc_id, score, snippet))

        self.display_result(doc_score_snippet)
        # return doc_score_snippet

    def get_score(self, doc_id, query_list):
        """
        compute accumulated score for terms in a document
        :return: the final score for a document
        """
        score = 0
        for term in query_list:
            term_obj = self.term_postings[term]
            term_postings = term_obj.get_postings()
            score += term_obj.get_df() * term_postings[doc_id]
        return score

    def get_snippet(self, doc_id, terms):
        """
        find snippets
        the distance of terms in each snippets is less than 60 characters
        or this two terms would be in two different snippets
        so each snippet stored would not overlap with each other
        :param terms: a list of querying term
        :param doc_id: doc's id
        :return: a list that contains snippets in different position
        """
        doc_dic = self.doc_data[doc_id]
        doc_text = ('<title>'+doc_dic['title']+'<\\title> ' + doc_dic['text']).lower()

        # use regular expression to highlight the terms and then find hits' indices
        pattern = '('+'|'.join(terms)+')'
        p = re.compile(pattern)
        doc_text = re.sub(r'\n', '', doc_text)
        doc_text = p.sub(r'<em>\1<\em>', doc_text)

        hit_indices = []
        for m in p.finditer(doc_text):
            hit_indices.append(m.start())

        snippets = []
        i = 0
        while i < len(hit_indices):
            j = i + 1
            while j < len(hit_indices) and hit_indices[j] - hit_indices[j - 1] < 60:
                j += 1
            # test on whether reach the beginning or end of the document
            left = 0 if hit_indices[i] - 30 < 0 else hit_indices[i] - 30
            right = len(doc_text) if hit_indices[j-1] + 30 > len(doc_text) else hit_indices[j-1] + 30
            # substring the text
            snippet = doc_text[left:right]
            snippets.append(snippet)
            i = j
        return snippets

    @staticmethod
    def dic_intersect(postings1, postings2):
        """
        find intersection between two dictionaries using their keys
        :param postings1: a dic
        :param postings2: a dic
        :return: the intersect dictionary
        """
        result = {}
        for posting in postings1.iterkeys():
            if posting in postings2:
                result[posting] = 1
        return result

    def display_result(self, doc_score_snippets):
        """
        display the search result, including total hits, rank, score etc.
        :param doc_score_snippets: a tuple: doc_id, score, snippet: a string list of snippets
        :return:
        """
        sorted_docs = sorted(doc_score_snippets, key=itemgetter(1), reverse=True)
        print len(self.term_postings)
        print 'Total number of hits: ', len(sorted_docs)
        print '--------------------'
        rank = 1
        for doc_id, score, snippets in sorted_docs:
            doc_dic = self.doc_data[doc_id]
            print 'Rank:\t', rank,
            print '\tScore:\t%.3f' % score
            print 'Title:\t', doc_dic['title']
            print 'Author:\t%s' % ', '.join(doc_dic['authors'])
            print 'snippets: \n\t%s' % '\n\t'.join(snippets)
            print '--------------------\n'
            rank += 1
            if rank > 10:
                break

if __name__ == "__main__":
    BS = DatabaseBuilder('wiki_all.txt')
    BS.build_database(from_file=True)
    SE = SearchEngine(database=BS)
    print 'Put it terms separated by space, use exit() to exit'
    while True:
        query = raw_input("Your query: ")
        if query == 'exit()':
            break
        SE.search(query.strip())