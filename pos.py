#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pattern.de import parsetree
import json
import re
from collections import Counter
from collections import defaultdict
import numpy as np
import math

#hint: this needs python2 and does not work well
# although it's for german, it seems to bring no good results
# e.g.: Rindfleisch, Tomate, Frischkäse tagged as JJ (adjective)

URL_KEY = 'url'
ING_KEY = 'ingredients'
TITLE_KEY = 'title'

# will also display the count of different ingredients
def apply_threshold_to_counter(counter, threshold):
    counter_thresh = dict(counter)  # apply threshold
    for key, count in counter.items():
        if count < threshold:
            del counter_thresh[key]
    print 'threshold:', threshold, 'different ingredients:', len(counter_thresh.keys())
    return counter_thresh


def sanitize(ingredients_raw):
    stemmed = []
    # stemmer = GermanStemmer()
    #stop_words = set(stopwords.words('german'))

    for ing in ingredients_raw:
        # remove braces, replace every other char (e.g. comma) with space
        # why space? picture this: "Salz,Pfeffer" we want this to be "Salz Pfeffer" for our next step
        # the tokenizer doesn't care about two spaces
        san = re.sub(r'\([^)]*\)', '', ing)
        san = re.sub('[^\w]+', ' ', san, flags=re.UNICODE)

        #s = parsetree(san)
        for word in san.split():
            w = word.decode()
            if w.istitle():
                stemmed.append(w)
            else:
                print 'discarded', w
        for sentence in s:
            for w in sentence.words:
                if w.type.startswith('N') or w.type is 'FW':
                    stemmed.append(w.string)  # stemmer.stem(word))
                else:
                    print ('discarded', w.string, w.type)
    return stemmed


if __name__ == '__main__':
    recipes_dict = {}
    ingredient_list = []
    s = 'Mir geht es gut und dir?'
    s = parsetree(s, relations=True, lemmata=True)

    recipes_dict = {}
    ingredient_list = []

    # read data (fairly simple, could be faster - but enough for now)
    with open('results.jl') as file:
        for row in file:
            recipe = json.loads(row)
            ingredients = recipe[ING_KEY]
            recipes_dict[recipe[URL_KEY]] = {TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients}
            ingredient_list.extend(ingredients)

    # analyze information
    counter = Counter(ingredient_list)
    print "different ingredients without threshold of recipes it appears in:", len(counter.keys())
    print '10 most common: ', counter.most_common(20)
    print 'now, apply threshold for minimum recipes the ingredients have to occur in'
    apply_threshold_to_counter(counter, 20)
    apply_threshold_to_counter(counter, 50)
    apply_threshold_to_counter(counter, 100)
    used = apply_threshold_to_counter(counter, 1000)
)

    print "now stem"

    counter = Counter(sanitize(ingredient_list)))
    print "different ingredients without threshold of recipes it appears in:", len(counter.keys())
    print '10 most common: ', counter.most_common(20)
    print 'now, apply threshold for minimum recipes the ingredients have to occur in'
    apply_threshold_to_counter(counter, 20)
    apply_threshold_to_counter(counter, 50)
    apply_threshold_to_counter(counter, 100)
    apply_threshold_to_counter(counter, 1000)
