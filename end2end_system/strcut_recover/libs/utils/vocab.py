class TypeVocab:
    key_words = [
        'title',
        'author',
        'affili',
        'mail',
        'foot',
        'fnote',
        'sec',
        'line',
        'fstline',
    ]
    struct_words = [
        'sec',
        'line',
        'fstline',
    ]

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word

        self.title_id  = self._words_ids_map['title']
        self.author_id = self._words_ids_map['author']
        self.affili_id = self._words_ids_map['affili']
        self.mail_id   = self._words_ids_map['mail']
        self.foot_id   = self._words_ids_map['foot']
        self.fnote_id  = self._words_ids_map['fnote']
        self.sec_id    = self._words_ids_map['sec']
        self.line_id   = self._words_ids_map['line']
        self.fstline_id   = self._words_ids_map['fstline']
        self.struct_word_ids = self.words_to_ids(self.struct_words)
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]


class RelationVocab: # This is for relations between lines and between secs
    key_words = [
        'contain', 
        'connect',
        'equality'
    ]

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word
        
        self.contain_id  = self._words_ids_map['contain']
        self.connect_id  = self._words_ids_map['connect']
        self.equality_id = self._words_ids_map['equality']
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]