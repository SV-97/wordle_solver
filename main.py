from itertools import count
from string import ascii_letters, ascii_lowercase
import numpy as np
from enum import Enum, auto
from collections import Counter
from fractions import Fraction
from math import log2
from random import choice, seed


WORD_LENGTH = 5
DICTIONARY_FILE = r"/usr/share/dict/american-english"


with open(DICTIONARY_FILE, "r") as f:
    words = [word.strip() for word in f.readlines()
             if len(word.strip()) == WORD_LENGTH]
    WORDS = sorted({word.lower() for word in words if all(
        letter in ascii_letters for letter in word)})
    WORDS_ARR = np.array([[*word] for word in WORDS])
    WORDS_ARR_NUM = np.vectorize(ord)(WORDS_ARR) - ord("a")


def wordle(word_to_guess):
    user_guess = None
    for i in range(5):
        user_guess = yield word_to_guess == user_guess


class Guess(Enum):
    CORRECT = auto()  # green
    SEMI_CORRECT = auto()  # yellow
    WRONG = auto()  # grey


class Wordle():
    def __init__(self, word_to_guess):
        self.word_to_guess = word_to_guess
        self.correct_distribution = Counter(word_to_guess)

    def __str__(self):
        return f"Wordle({self.word_to_guess})"

    __repr__ = __str__

    def guess(self, word):
        # assert(len(word) == WORD_LENGTH)
        # assert(word in WORDS)
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
    guess_correct_count = Counter([letter for letter, r in zip(
        guess, result) if r is Guess.CORRECT or r is Guess.SEMI_CORRECT])

    def check_if_word_remains(word):
        non_correct_word = {letter for letter, count in Counter(
            word).items() if count != guess_correct_count[letter]}
        if (word == guess).all():  # remove the just-guessed word
            return False
        for i, (guess_letter, word_letter, r) in enumerate(zip(guess, word, result)):
            if r is Guess.CORRECT:
                known[i] = guess_letter
                if word_letter != guess_letter:
                    return False
            elif r is Guess.WRONG:
                if guess_letter in non_correct_word:
                    return False
        return True

    mask = np.zeros(words.shape[0], dtype=np.bool8)
    for i, word in enumerate(words):
        mask[i] = check_if_word_remains(word)
    return words[mask, :]


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
    masked_words = words[:, mask]
    letter_counts = Counter({ord(l) - ord("a"): 0 for l in ascii_lowercase})
    letter_counts.update(masked_words.flatten())
    c1 = sorted(letter_counts.items(), key=lambda x: x[0])
    counts = np.array([x[1] for x in c1], dtype=np.float32)
    letter_probability = counts / (words.shape[0] * mask.sum())
    letter_information = -letter_probability * np.log2(letter_probability)

    # np.vectorize(ord)(masked_words) - ord("a")
    masked_letter_index = masked_words
    word_entropy = letter_information[masked_letter_index].sum(
        axis=1) * letter_probability[masked_letter_index].sum(axis=1) / mask.sum()

    idx = np.argmax(word_entropy)
    return words[idx, :], word_entropy[idx]


def solver(game):
    """
    We know how common each letter is. Based on this we can assign each letter a probability.
    This means we can assign each letter some information - and via this we assign a word its
    information content as the sum of the letter's information contents. We furthermore assign
    each word a probability based on it's letter's probabilities. Given this we can find the word
    with the highest expected information content among all words that are viable solutions.
    This is the word we'll choose next.
    """
    words = WORDS_ARR_NUM
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


seed(0)
right = 0
for i in count(1):
    word = choice(WORDS_ARR_NUM)
    game = Wordle(word)
    # print(game)
    # print(solver(game))
    tries, guess = solver(game)
    if tries <= 6 and (word == guess).all():
        right += 1
    print(right/i)
