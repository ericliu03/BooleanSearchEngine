__author__ = 'EricLiu'

import re
import time
import shelve
from json import load
from math import log10, sqrt
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
    def __init__(self, raw_file=None):
        """
        Initialize the batabase builder, if raw file, which is the wiki file, is not
        provided, it will read the data from database file.
        :param raw_file: the wikipedia file
        :return: None
        """
        from_file = True
        if raw_file is not None:
            # if raw_file is not provided, the created database will be read
            self.doc_dics = self.get_info_from_file(raw_file)
            # for computing document frequency
            self.doc_term_dic = {}
            from_file = False

        # for parsing the text, used also in search engine
        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')

        # for store or read data from file
        self.bs_data = shelve.open('BooleanSearch.db')
        self.doc_data = {}
        self.term_postings = {}
        start_time = time.time()
        self.build_database(from_file=from_file)
        print 'Total time used: %.2f sec' % (time.time() - start_time)

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
                # process display
                if doc_id % 80 == 0:
                    print '%0.2f%%' % (100.0 * doc_id/doc_length)
                doc_id += 1
                # process the document and add to term-posting list
                doc_detail = doc_dic['title']+' '+doc_dic['text']
                processed_doc = self.pre_process(doc_detail)
                self.build_term_dic(doc_id, processed_doc)
                self.build_doc_dic(doc_id, doc_dic, processed_doc)
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

    def build_term_dic(self, doc_id, terms):
        """
        compute term frequency and store them to term-postings list
        for each term appears in a doc
        @part II revision: I used to use 'if term in self.term_postings.iterkeys()
        to check whether the term exists or not, apparently a mistake approach
        :param doc_id: document ID
        :param terms: a list of terms from parsed document text
        :return: None
        """
        # for computing term frequency
        counter = Counter(terms)

        # store postings and tf for each term
        for term, raw_tf in counter.iteritems():

            if term in self.term_postings:
                term_obj = self.term_postings[term]
            else:
                term_obj = Term()
                self.term_postings[term] = term_obj

            term_obj.add_posting(doc_id, raw_tf)

    def build_doc_dic(self, doc_id, doc_dic, terms):
        """
        build the document dictionary for displaying searching result
        the length of the text is stored for computing tf in each document
        @Part II addition: compute the doc's length and store it in the doc dictionary
        @bug fixed: used to sqrt twice
        :param doc_id: document ID
        :param doc_dic: the original document dictionary
        :param terms: a list contains terms from processed document
        :return: None
        """
        doc_length = 0
        counter = Counter(terms)
        for raw_tf in counter.itervalues():
            weight = 1 + log10(raw_tf)
            doc_length += weight * weight

        doc_length = sqrt(doc_length)
        temp_dic = {
            'title': doc_dic['title'],
            'authors': doc_dic['authors'],
            'text': doc_dic['text'],
            'length': doc_length
        }
        self.doc_data[doc_id] = temp_dic

    def compute_df(self):
        """
        compute the document frequency for each term
        :return: None
        """
        for term in self.term_postings.itervalues():
            raw_df = len(term.get_postings())
            term.set_df(raw_df)

    @staticmethod
    def get_info_from_file(file_name):
        """
        load json from a file to python objective
        here the docs is a dictionary, keys are document number and values
        are dictionaries in which keys are 'title', 'authors' and etc.
        """
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
        self.raw_df = 0
        self.postings = {}

    def set_df(self, raw_df):
        self.raw_df = raw_df

    def add_posting(self, doc_id, raw_tf):
        self.postings[doc_id] = raw_tf

    def get_postings(self):
        return self.postings.copy()

    def get_df(self):
        return self.raw_df


class SearchEngine:
    """
    A boolean search engine provided ranking and query from given database
    """
    def __init__(self, database=None):
        """
        If database is not provided, it will create a database from file 'wiki_all.txt'
        :param database: a DatabaseBuilder Class Object
        :return:
        """
        if database is None:
            database = DatabaseBuilder('wiki_all.txt')

        self.term_postings = database.term_postings
        self.pre_process = database.pre_process
        self.doc_data = database.doc_data

    def search(self, query_string, score_com='vec'):
        """
        find the intersection of input term's postings, get idf
        from the postings, find the word's tf and snippets
        compute the accumulated score, and display the result
        @Part II addition: add Vector space model to compute score
        :param score_com: the way of computing ranking score: 'vec' or 'raw'
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
            score = self.get_score(doc_id, query_list, score_com)
            snippet = self.get_snippet(doc_id, query_list)
            doc_score_snippet.append((doc_id, score, snippet))

        self.display_result(doc_score_snippet)
        # return doc_score_snippet

    def get_score(self, doc_id, query_list, score_com='vec'):
        """
        compute accumulated score for terms in a document
        @Part II addition: add Vector space model using cosine distance
        :param doc_id: document ID
        :param query_list: a list of terms from preprocessed query
        :param score_com: the way of computing ranking score
        :return:the final score for a document
        """
        score = 0
        query_tf_dic = Counter(query_list)
        num_doc = len(self.doc_data)
        doc_length = self.doc_data[doc_id]['length']

        for term, query_raw_tf in query_tf_dic.iteritems():
            term_obj = self.term_postings[term]
            term_postings = term_obj.get_postings()
            if score_com == 'raw':
                # using raw tf to compute score
                score += term_postings[doc_id]
            elif score_com == 'vec':
                # using vector space model
                tf = 1 + log10(query_raw_tf)
                idf = log10(1.0 * num_doc / term_obj.get_df())
                query_weight = tf * idf
                doc_weight = (1 + log10(1.0 * term_postings[doc_id])) / doc_length
                score += query_weight * doc_weight
            elif score_com == 'tfidf':
                tf = 1 + log10(1.0 * term_postings[doc_id])
                idf = log10(1.0 * len(self.doc_data) / term_obj.get_df())
                score += tf * idf
            else:
                raise ValueError("the way of computing score is either 'vec' or 'raw'")

        return score

    def get_snippet(self, doc_id, terms):
        """
        find snippets
        the distance of terms in each snippets is less than 60 characters
        or this two terms would be in two different snippets
        so each snippet stored would not overlap with each other
        @Part II revision: add a counter for counting the number of snippets, only first 5 will remain
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
        count_snip = 0
        left_ptr = 0
        while left_ptr < len(hit_indices):
            # only first 5 snippets
            if count_snip >= 5:
                break
            else:
                count_snip += 1

            right_ptr = left_ptr + 1
            while right_ptr < len(hit_indices) and hit_indices[right_ptr] - hit_indices[right_ptr - 1] < 60:
                right_ptr += 1
            # test on whether reach the beginning or end of the document
            left = 0 if hit_indices[left_ptr] - 30 < 0 else hit_indices[left_ptr] - 30
            right = len(doc_text) if hit_indices[right_ptr-1] + 30 > len(doc_text) else hit_indices[right_ptr-1] + 30
            # substring the text
            snippet = doc_text[left:right]
            snippets.append(snippet)
            left_ptr = right_ptr
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
        :return: None
        """
        sorted_docs = sorted(doc_score_snippets, key=itemgetter(1), reverse=True)
        print 'Total number of hits:\t', len(sorted_docs)
        print '--------------------'
        rank = 1
        for doc_id, score, snippets in sorted_docs:
            doc_dic = self.doc_data[doc_id]
            print 'Rank:\t', rank
            print 'Score:\t%.3f' % score
            print 'Title:\t', doc_dic['title']
            print 'Author:\t', doc_dic['authors']
            print 'snippets: \n\t%s' % '\n\t'.join(snippets)
            print '\n--------------------'
            rank += 1
            if rank > 10:
                break

if __name__ == "__main__":
    # if you want to build a new database from json file, add a parameter to next line
    BS = DatabaseBuilder()
    SE = SearchEngine(database=BS)

    print 'Put in terms separated by space, use exit() to exit.'
    while True:
        query_type = raw_input("Your query type (raw or vec): ")
        if query_type == 'exit()':
            break
        while not (query_type == 'vec' or query_type == 'raw' or query_type == 'tfidf'):
            query_type = raw_input("only 'vec' or 'raw' could be accepted: ")
        query = raw_input("Your query: ")

        SE.search(query.strip(), score_com=query_type)