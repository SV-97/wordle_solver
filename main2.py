DICTIONARY_FILE = r"/usr/share/dict/american-english"
from concurrent.futures import process
from itertools import count
from string import ascii_letters
import numpy as np
from enum import Enum, auto
from collections import Counter
from fractions import Fraction
from math import log2
from random import choice

WORD_LENGTH = 5


with open(DICTIONARY_FILE, "r") as f:
    words = [word.strip() for word in f.readlines() if len(word.strip()) == WORD_LENGTH]
    WORDS = sorted({word.lower() for word in words if all(letter in ascii_letters for letter in word)})


def wordle(word_to_guess):
    user_guess = None
    for i in range(5):
        user_guess = yield word_to_guess == user_guess 


class Guess(Enum):
    CORRECT = auto() # green
    SEMI_CORRECT = auto() # yellow
    WRONG = auto() # grey


class Wordle():
    def __init__(self, word_to_guess):
        self.word_to_guess = word_to_guess
        self.correct_distribution = Counter(word_to_guess)

    def __str__(self):
        return f"Wordle({self.word_to_guess})"

    __repr__ = __str__

    def guess(self, word):
        assert(len(word) == WORD_LENGTH)
        assert(word in WORDS)
        response = [Guess.WRONG] * len(word)
        counter = self.correct_distribution.copy()
        for i, (guess, correct) in enumerate(zip(word, self.word_to_guess)):
            if guess == correct:
                counter[guess] -= 1
                response[i] = Guess.CORRECT
        for i, (guess, correct) in enumerate(zip(word, self.word_to_guess)):
            if response[i] is Guess.CORRECT:
                continue
            if guess in counter:
                if counter[guess] > 0:
                    counter[guess] -= 1
                    response[i] = Guess.SEMI_CORRECT
        return response


def filter_new_words(words, guess, result, known):
    new_words = []
    guess_correct_count = Counter([letter for letter, r in zip(guess, result) if r is Guess.CORRECT or r is Guess.SEMI_CORRECT])
    def process(word):
        non_correct_word = {letter for letter, count in Counter(word).items() if count != guess_correct_count[letter]}
        if word == guess: # remove the just-guessed word
            # print("remove ", word, " by equality")
            return
        for i, (guess_letter, word_letter, r) in enumerate(zip(guess, word, result)):
            if r is Guess.CORRECT:
                known[i] = guess_letter
                if word_letter != guess_letter:
                    #print("remove ", word, " wrong letter: expected ", guess_letter, "got ", word_letter)
                    return
            elif r is Guess.WRONG:
                if guess_letter in non_correct_word:
                    #print("remove ", word, " contains wrong letter ", guess_letter)
                    return
        new_words.append(word)
    for word in words:
        # print(word)
        process(word)
    # print(new_words)
    return new_words



def solver(game):
    """
    We know how common each letter is. Based on this we can assign each letter a probability.
    This means we can assign each letter some information - and via this we assign a word its
    information content as the sum of the letter's information contents. We furthermore assign
    each word a probability based on it's letter's probabilities. Given this we can find the word
    with the highest expected information content among all words that are viable solutions.
    This is the word we'll choose next.
    """
    def word_of_maximal_entropy(words, known):
        """
        Via the `known` parameter we select only those positions in our word that we don't already know,
        because these are the only ones where we can still resolve uncertainty.
        KNOWN ISSUE:
        This function currently disregards the fact that a word like eeeee is not actually probable just
        because e is a common letter we should probably (hehe) improve word_probability by using further
        information about the languages structure.
        """
        mask = np.array([k is None for k in known])
        mask = np.ones(mask.shape, dtype=np.bool8)
        masked_words = np.array([[*word] for word in words])[:, mask]
        letter_counts = Counter(masked_words.flatten())
        letter_probability = {letter : Fraction(n, int(mask.sum()) * len(words)) for letter, n in letter_counts.items()}
        letter_information = {l : -p * log2(p) for l, p in letter_probability.items()}
        
        @np.vectorize
        def get_letter_information(letter):
            return letter_information[letter]

        @np.vectorize
        def get_letter_probability(letter):
            return letter_probability[letter]

        word_entropy = get_letter_information(masked_words).sum(axis=1) * get_letter_probability(masked_words).sum(axis=1) / float(mask.sum())
        idx = np.argmax(word_entropy)
        return words[idx], word_entropy[idx]
    words = WORDS
    known = [None] * WORD_LENGTH
    for i in range(10):
        # print(f"{known=}")
        if all(k is not None for k in known):
            guess = "".join(known)
            break
        guess, entropy = word_of_maximal_entropy(words, known)
        result = game.guess(guess)
        # print(guess, entropy, result)
        if all(r is Guess.CORRECT for r in result):
            break
        else:
            words = filter_new_words(words, guess, result, known)
    return i, guess

right = 0
for i in count(1):
    game = Wordle(choice(WORDS))
    # print(game)
    # print(solver(game))
    tries, guess = solver(game)
    if tries <= 6:
        right += 1
    print(right/i)
# print(game, game.guess("baron"))
