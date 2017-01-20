#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import collections
import numpy as np
from collections import Counter
from collections import defaultdict

URL_KEY = 'url'
ING_KEY = 'ingredients'
TITLE_KEY = 'title'

# will also display the count of different ingredients
def apply_threshold_to_counter(counter, threshold):
	counter_thresh = dict(counter) # apply threshold
	for key, count in counter.items():
		if count < threshold:
			del counter_thresh[key]
	print('threshold:', threshold, 'different ingredients:', len(counter_thresh.keys()))
	return counter_thresh


# this will build the json the d3js-html then will read
# has a "strange' json structure, don't ask me who thought of that....
# further info: https://d3js.org/ or talk to thorsten for now :-P
def build_bubble_json(used):
	# build bubble chart json structure
	bubble_dict = defaultdict(list)
	for key, value in used.items():
		parent_item = {}
		item = {}
		parent_item['children'] = []
		parent_item['name'] = key
		parent_item['children'].append(item)

		item['name'] = key
		item['size'] = value
		item['url'] = ''

		bubble_dict['children'].append(parent_item)

	with open('bubble.json', 'wt') as outfile:
		json.dump(bubble_dict, outfile, sort_keys=True)


if __name__ == '__main__':
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
	print("different ingredients without threshold of recipes it appears in:", len(counter.keys()))
	print('10 most common: ', counter.most_common(10))
	print('now, apply threshold for minimum recipes the ingredients have to occur in')
	apply_threshold_to_counter(counter, 20)
	apply_threshold_to_counter(counter, 50)
	apply_threshold_to_counter(counter, 100)
	used = apply_threshold_to_counter(counter, 1000)

	# first visualization
	build_bubble_json(used)
