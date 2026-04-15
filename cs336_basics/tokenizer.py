import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
    
    def encode(self, text):

        pass
    
    def decode(self, ids):
        pass

    def bpe_train(input_path, vocab_size, special_tokens):
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for token in special_tokens:
            vocab[next_id] = token.encode("utf-8")
            next_id += 1

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        #split by special tokens first
        sptok = "("+ "|".join(map(re.escape, special_tokens)) + ")"
        chunks = re.split(sptok, text)

        #split by PAT
        word_freq = Counter()
        for chunk in chunks:
            for tok in re.finditer(PAT, chunk):
                byte_tuple = tuple(tok.group().encode("utf-8"))
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
                        new_word.append(word[i:i+2])  # merged
                        i += 2
                    else:
                        new_word.append((word[i],))
                        i += 1
                #remove tuple to change into regular list
                new_word = tuple(sum(new_word, ()))
                new_word_freq[new_word] = freq
            word_freq = new_word_freq
            #add pair to vocab and increment id
            vocab[next_id] = bytes(best_pair)
            next_id += 1
        return vocab, merges






        



        


