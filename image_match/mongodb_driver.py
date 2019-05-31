import logging
from multiprocessing import Process, Queue
from multiprocessing.managers import queue as managerQueue

import numpy as np
import pymongo

from .signature_database_base import SignatureDatabaseBase
from .signature_database_base import normalized_distance

logger = logging.getLogger(__name__)


class SignatureMongo(SignatureDatabaseBase):
    """MongoDB driver for image-match

    """

    def __init__(self, collection, best_count_of_result=20, count_must_equal_n=True, *args, **kwargs):
        """Additional MongoDB setup

        Args:
            collection (collection): a MongoDB collection instance
            args (Optional): Variable length argument list to pass to base constructor
            kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        Examples:
            >>> from image_match.mongodb_driver import SignatureMongo
            >>> from pymongo import MongoClient
            >>> client = MongoClient(connect=False)
            >>> c = client.images.images
            >>> ses = SignatureMongo(c)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.collection = collection
        # Extract index fields, if any exist yet
        if self.collection.count() > 0:
            self.index_names = [field for field in self.collection.find_one({}).keys()
                                if field.find('simple') > -1]

        super(SignatureMongo, self).__init__(*args, **kwargs)

        self.mongo_args = None

        self.best_count_of_result = best_count_of_result
        self.count_must_equal_n = count_must_equal_n

    def insert_single_record(self, rec, refresh_after=False):
        self.collection.insert(rec)

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def index_collection(self):
        """Index a collection on words.

        """
        # Index on words
        self.index_names = [field for field in self.collection.find_one({}).keys()
                            if field.find('simple') > -1]
        for name in self.index_names:
            self.collection.create_index(name)

    def _set_mongo_client_args(self):
        try:
            if self.mongo_args is None:
                self.mongo_args = CollectionArgs()
                self.mongo_args.collection = self.collection.name
                self.mongo_args.database = self.collection.database.name

                _client = self.collection.database.client

                self.mongo_args.client_host = _client.client_args["host"]
                self.mongo_args.client_port = _client.client_args["port"]
                self.mongo_args.client_document_class = _client.client_args["document_class"]
                self.mongo_args.client_kwargs = _client.client_args["kwargs"]
                self.mongo_args.client_connect = _client.client_args["connect"]
                self.mongo_args.client_tz_aware = _client.client_args["tz_aware"]
        except Exception as e:
            self.logger.error(e, exc_info=1)
            self.mongo_args = None

    def search_single_record(self, rec, n_parallel_words=1, word_limit=None,
                             process_timeout=None, maximum_matches=1000, filter=None):
        if n_parallel_words is None or n_parallel_words > 1:
            return self._search_single_record_with_multi_processing(rec, n_parallel_words, word_limit,
                                                                    process_timeout, maximum_matches)

        return self._search_single_record_with_current_processing(rec, word_limit, maximum_matches)

    def _search_single_record_with_multi_processing(self, rec, n_parallel_words, word_limit, process_timeout,
                                                    maximum_matches):
        if word_limit is None:
            word_limit = self.N

        initial_q = managerQueue.Queue()

        [initial_q.put({field_name: rec[field_name]}) for field_name in self.index_names[:word_limit]]

        # enqueue a sentinel value so we know we have reached the end of the queue
        initial_q.put('STOP')
        queue_empty = False

        # create an empty queue for results
        results_q = Queue()

        # create a set of unique results, using MongoDB _id field
        unique_results = set()

        # init mongo collection args
        self._set_mongo_client_args()
        _mongo_args = self.mongo_args or self.collection

        sorted_queue = SortedQuene()

        while True:

            # build children processes, taking cursors from in_process queue first, then initial queue
            p = list()
            while len(p) < n_parallel_words:
                word_pair = initial_q.get()
                if word_pair == 'STOP':
                    # if we reach the sentinel value, set the flag and stop queuing processes
                    queue_empty = True
                    break
                if not initial_q.empty():
                    p.append(Process(target=get_next_match,
                                     args=(results_q,
                                           word_pair,
                                           _mongo_args,
                                           np.array(rec['signature']),
                                           self.distance_cutoff,
                                           maximum_matches)))

            if len(p) > 0:
                for process in p:
                    process.start()
            else:
                break

            # collect results, taking care not to return the same result twice

            num_processes = len(p)

            while num_processes:
                results = results_q.get()
                if results == 'STOP':
                    num_processes -= 1
                else:
                    key = results["id"]
                    if key not in unique_results:
                        unique_results.add(key)
                        sorted_queue.append(results["dist"], results)

            for process in p:
                process.join()

            # yield a set of results
            if queue_empty:
                break

        return sorted_queue.pop_asc_list(top_n=self.best_count_of_result, count_must_equal_n=self.count_must_equal_n)

    def _search_single_record_with_current_processing(self, rec, word_limit, maximum_matches):
        if word_limit is None:
            word_limit = self.N

        field_list = ({field_name: rec[field_name]} for field_name in self.index_names[:word_limit])

        # create a set of unique results, using MongoDB _id field
        unique_results = set()

        sorted_queue = SortedQuene()

        for word_pair in field_list:
            try:
                result_list = get_next_match_in_current_processing(
                    word_pair,
                    self.collection,
                    np.array(rec['signature']),
                    self.distance_cutoff,
                    maximum_matches
                )
                if result_list:
                    for results in result_list:
                        key = results["id"]
                        if key not in unique_results:
                            unique_results.add(key)
                            sorted_queue.append(results["dist"], results)

            except Exception as e:
                self.logger.error(e, exc_info=1)

        return sorted_queue.pop_asc_list(top_n=self.best_count_of_result, count_must_equal_n=self.count_must_equal_n)


class CollectionArgs(object):
    __slots__ = ("client_host", "client_port", "client_document_class", "client_tz_aware",
                 "client_connect", "client_kwargs", "database", "collection")

    def __init__(self):
        self.client_host = None
        self.client_port = None
        self.client_document_class = None
        self.client_tz_aware = None
        self.client_connect = None
        self.client_kwargs = None
        self.database = ""
        self.collection = ""


class SortedQuene(object):
    __slots__ = ("_dict", "_score_set")

    def __init__(self):
        self._dict = {}
        self._score_set = set()

    def append(self, score, obj):
        # 添加
        if score not in self._dict:
            self._dict[score] = []

        self._dict[score].append(obj)

        # 排序
        self._score_set.add(score)

    def pop_asc_list(self, top_n=100, count_must_equal_n=True):
        result = []
        _score_list = list(self._score_set)
        _score_list.sort()
        for _ in range(top_n):
            if not _score_list:
                break

            score = _score_list.pop(0)
            if score in self._score_set:
                self._score_set.remove(score)
            result.extend(self._dict.pop(score))

            if len(result) >= top_n:
                break

        if count_must_equal_n and len(result) >= top_n:
            return result[:top_n]

        return result


def get_next_match(result_q, word, collection_args, signature, cutoff=0.5, max_in_cursor=100):
    """Given a cursor, iterate through matches

    Scans a cursor for word matches below a distance threshold.
    Exhausts a cursor, possibly enqueuing many matches
    Note that placing this function outside the SignatureCollection
    class breaks encapsulation.  This is done for compatibility with
    multiprocessing.

    Args:
        result_q (multiprocessing.Queue): a multiprocessing queue in which to queue results
        word (dict): {word_name: word_value} dict to scan against
        collection_args (CollectionArgs|collection): pymongo collection args or a pymongo collection
        signature (numpy.ndarray): signature array to match against
        cutoff (Optional[float]): normalized distance limit (default 0.5)
        max_in_cursor (Optional[int]): if more than max_in_cursor matches are in the cursor,
            ignore this cursor; this column is not discriminatory (default 100)

    """
    client = None
    try:
        if isinstance(collection_args, CollectionArgs):
            client = pymongo.MongoClient(
                host=collection_args.client_host,
                port=collection_args.client_port,
                document_class=collection_args.client_document_class,
                tz_aware=collection_args.client_tz_aware,
                connect=collection_args.client_connect,
                **collection_args.client_kwargs)
            collection = client[collection_args.database][collection_args.collection]
        else:
            collection = collection_args

        curs = collection.find(word, projection=['_id', 'signature', 'path', 'metadata'])

        # if the cursor has many matches, then it's probably not a huge help. Get the next one.
        if curs.count() > max_in_cursor:
            result_q.put('STOP')
            return

        while True:
            try:
                rec = curs.next()
                dist = normalized_distance(np.reshape(signature, (1, signature.size)), np.array(rec['signature']))[0]
                result_q.put({'dist': dist, 'path': rec['path'], 'id': rec['_id'],
                              'metadata': rec.get('metadata', {})})
            except StopIteration:
                # do nothing...the cursor is exhausted
                break
        result_q.put('STOP')
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                logger.error(e, exc_info=1)


def get_next_match_in_current_processing(word, collection_args, signature, cutoff=0.5, max_in_cursor=100):
    """Given a cursor, iterate through matches

    Scans a cursor for word matches below a distance threshold.
    Exhausts a cursor, possibly enqueuing many matches
    Note that placing this function outside the SignatureCollection
    class breaks encapsulation.  This is done for compatibility with
    multiprocessing.

    Args:
        word (dict): {word_name: word_value} dict to scan against
        collection_args (collection): pymongo collection args or a pymongo collection
        signature (numpy.ndarray): signature array to match against
        cutoff (Optional[float]): normalized distance limit (default 0.5)
        max_in_cursor (Optional[int]): if more than max_in_cursor matches are in the cursor,
            ignore this cursor; this column is not discriminatory (default 100)

    """
    result = []
    collection = collection_args

    curs = collection.find(word, projection=['_id', 'signature', 'path', 'metadata'])

    # if the cursor has many matches, then it's probably not a huge help. Get the next one.
    if curs.count() > max_in_cursor:
        return result

    while True:
        try:
            rec = curs.next()
            dist = normalized_distance(np.reshape(signature, (1, signature.size)), np.array(rec['signature']))[0]
            result.append({'dist': dist, 'path': rec['path'], 'id': rec['_id'], 'metadata': rec.get('metadata', {})})
        except StopIteration:
            # do nothing...the cursor is exhausted
            break

    return result
