import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []
        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        tokens = []
        
        if not self.special_tokens:
            chunks = [text]
        else:
            sptok = "("+ "|".join(map(re.escape, self.special_tokens)) + ")"
            chunks = re.split(sptok, text)

        for chunk in chunks:
            if chunk in self.special_tokens:
                tokens.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue

            parts = re.findall(PAT, chunk)

            for part in parts:
                word = [bytes([x]) for x in part.encode("utf-8")]

                for merge in self.merges:
                    i = 0
                    new_word = []
                    while i < len(word):
                        if i < len(word)-1 and (word[i], word[i+1]) == merge:
                            new_word.append(merge[0]+ merge[1])
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word

                for w in word:
                    tokens.append(self.bytes_to_id[w])

        return tokens
        
    
    def decode(self, ids):
        
        byte_seq = b"".join(self.id_to_bytes[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")


def bpe_train(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    #split by special tokens first
    
    if not special_tokens:
        chunks = [text]
    else:
        sptok = "("+ "|".join(map(re.escape, special_tokens)) + ")"
        chunks = re.split(sptok, text)

    #split by PAT
    word_freq = Counter()
    for chunk in chunks:
        for tok in re.finditer(PAT, chunk):
            byte_tuple = tuple(bytes([x]) for x in tok.group().encode("utf-8"))

            word_freq[byte_tuple] += 1
    
    merges = []
    while len(vocab) < vocab_size:
        pair_counts = Counter()
        for word, freq in word_freq.items():
            #iterate through words, combine adjacent letters into pairs
            #increment respectiv pair counts for merge later
            for i in range(len(word) - 1):
                #combine into pairs
                pair = (word[i], word[i+1])
                pair_counts[pair] += freq
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0] #tie-break max

        merges.append(best_pair)
        new_word_freq = {}
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word): #merge best_pairs
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(word[i]+word[i+1])  # merged
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            #remove tuple to change into regular list
            #new_word = tuple(sum(new_word, ()))
            if new_word_freq[tuple(new_word)]:
                new_word_freq[tuple(new_word)] += freq
            else:
                new_word_freq[tuple(new_word)] = freq
        word_freq = new_word_freq
        #add pair to vocab and increment id
        vocab[next_id] = best_pair[0]+best_pair[1]
        next_id += 1
    return vocab, merges

    





        



        


