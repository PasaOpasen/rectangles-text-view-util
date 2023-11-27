[![PyPI version](https://badge.fury.io/py/ordered-rectangles.svg)](https://pypi.org/project/ordered-rectangles/)
[![Downloads](https://pepy.tech/badge/ordered-rectangles)](https://pepy.tech/project/ordered-rectangles)
[![Downloads](https://pepy.tech/badge/ordered-rectangles/month)](https://pepy.tech/project/ordered-rectangles)
[![Downloads](https://pepy.tech/badge/ordered-rectangles/week)](https://pepy.tech/project/ordered-rectangles)

# ordered_rectangles

This package is an utility purposed to:
* view float rectangles relations (relative positions and the order) in the text format
* change rectangles order by changing rectangles labels in the text format view file

Usual imports:
```python
from ordered_rectangles import OrderedRectangles, RectTextViewer, read_text, read_json, write_text, write_json
```

Usage example:
```python
class OrderedRectangles:
    """
    stores `(x1, y1, x2, y2)` numpy-ordered rectangles and provides an ability to
        1. show its text map which reflects the rectangles order and positions with different disretization level
        2. save/load the rectangles with/without the map to json file
        3. change the map manually to change the rectangles order easily
```

```python
# How to create an object:
>>> d = OrderedRectangles([(0.1, 0.2, 0.3, 0.4), (0.1, 0.6, 0.2, 1.1)])

# How to view the map with `units` discretization level:
>>> units = 15
>>> mp = d.get_order_map(units=units); print(mp) # doctest: +NORMALIZE_WHITESPACE
    1#### 2#######
    #   # #      #
    #   # ########
    #####

# Use `show_order_map` method to simplify this step:
>>> d.show_order_map(units=units, show_order=False) # doctest: +NORMALIZE_WHITESPACE
    ##### ########
    #   # #      #
    #   # ########
    #####

# Let's try bigger example:
>>> d = OrderedRectangles(
...     [
...         (0.1, 0.2, 0.3, 0.4), (0.1, 0.6, 0.2, 1.1),
...         (0.1, 1.2, 0.3, 1.4), (0.1, 1.5, 0.25, 2.3),
...         (0.35, 0.2, 0.6, 0.5), (0.4, 0.6, 0.6, 1.4)
...     ]
... )

# `units` <= 0 means to search the best value automatically
>>> d.show_order_map(units=0)  # doctest: +NORMALIZE_WHITESPACE
    1########      2###################  3######## 4##############################
    #       #      #                  #  #       # #                             #
    #       #      #                  #  #       # #                             #
    #       #      #                  #  #       # #                             #
    #       #      ####################  #       # #                             #
    #       #                            #       # #                             #
    #       #                            #       # ###############################
    #       #                            #       #
    #########                            #########
    5############
    #           #  6##############################
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #############  ###############################

# I can swap 5th and 6th rectangles programmatically:
>>> d[5], d[6] = d[6], d[5]
>>> d.show_order_map(units=0)  # doctest: +NORMALIZE_WHITESPACE
    1########      2###################  3######## 4##############################
    #       #      #                  #  #       # #                             #
    #       #      #                  #  #       # #                             #
    #       #      #                  #  #       # #                             #
    #       #      ####################  #       # #                             #
    #       #                            #       # #                             #
    #       #                            #       # ###############################
    #       #                            #       #
    #########                            #########
    6############
    #           #  5##############################
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #           #  #                             #
    #############  ###############################

# The key feature is that I can change this order by changing the string map:
>>> d3 = d[3]; d4 = d[4]
>>> s = d.get_order_map(units=0).translate({ord('4'): '3', ord('3'): '4'})
>>> d.load_order_map(s)
>>> assert d[3] == d4 and d[4] == d3

# However it's not efficiet to change the rectangles order by changing the string map programmatically.
# Instead, it is supposed that u will save the map to some text file, change numbers manually and load the map from
#     the file. `load_order_map` methods supports file path as an input too.

# Finally there is a way to save and load the rectangles as JSON:
>>> tempdir = 'tmp'
>>> jsdir = os.path.join(tempdir, 'separate-jsons'); mkdir(jsdir)
>>> mapdir = os.path.join(tempdir, 'separate-maps'); mkdir(mapdir)
>>> js_file = os.path.join(jsdir, 'rects.json')
>>> map_file = os.path.join(mapdir, 'rects.txt')
>>> d.to_json(path=js_file, save_map=map_file, units=0)
>>> dct = read_json(js_file); print(str(dct).replace(os.sep, '/').replace('//', '/'))
{'rects': [[0.1, 0.2, 0.3, 0.4], [0.1, 0.6, 0.2, 1.1], [0.1, 1.5, 0.25, 2.3], [0.1, 1.2, 0.3, 1.4], [0.4, 0.6, 0.6, 1.4], [0.35, 0.2, 0.6, 0.5]], 'map': '../separate-maps/rects.txt'}

# As u can see, tha path to maps file is saved as absolute or relative to the json file.
# U can change the order in the map manually and reload rectangles

>>> write_text(map_file, read_text(map_file).translate({ord('1'): '2', ord('2'): '1'}))
>>> d2 = OrderedRectangles.from_json(js_file)
>>> assert d[1] == d2[2] and d[2] == d2[1]
```

