
from typing import Union, Iterable, Tuple, List, Dict, TypeVar
from typing_extensions import TypeAlias

import os
from pathlib import Path
import json

import numpy as np

#region ALIASES

T = TypeVar('T')

PathLike: TypeAlias = Union[str, os.PathLike]

BoxFloat: TypeAlias = Tuple[float, float, float, float]
BoxInt: TypeAlias = Tuple[int, int, int, int]
Box: TypeAlias = Union[BoxInt, BoxFloat]

array2D: TypeAlias = np.ndarray
arrayRects: TypeAlias = array2D
"""2D array of rectangles with columns (x1, y1, x2, y2) where x is rows dim"""
arrayRectsInt: TypeAlias = arrayRects
"""
2D array with integer rectangles coordinates in form (x1, y1, x2, y2)

where (1, 2, 3, 4) means to fill [1:3, 2:4] including bounds, one-based
"""

#endregion

#region CONSTANTS

_EMPTY_FILLER_INT: int = -2
_BOUND_FILLER_INT: int = -1

FILLERS_INT = (_BOUND_FILLER_INT, _EMPTY_FILLER_INT)
"""values in the numpy map reserved for non-labeled data"""

EMPTY_FILLER_STR: str = ' '
"""fill value for empty space in the ordered map string view"""
BOUND_FILLER_STR: str = '#'
"""fill value for usual bound in the ordered map string view"""

UNITS_COUNT_SEARCH_MIN_VALUE: int = 4
"""min unit size available for autosearch algorithm"""

UNITS_COUNT_SEARCH_MAX_VALUE: int = 300
"""max unit size available for autosearch algorithm"""

#endregion


#region FUNCS

def mkdir_of_file(file_path: PathLike):
    """
    для этого файла создаёт папку, в которой он должен лежать
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def mkdir(dir: PathLike):
    """mkdir with parents"""
    Path(dir).mkdir(parents=True, exist_ok=True)


def read_json(path: PathLike):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: PathLike, data):
    mkdir_of_file(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def read_text(path: PathLike, encoding: str = 'utf-8'):
    return Path(path).read_text(encoding=encoding, errors='ignore')


def write_text(path: PathLike, text: str, encoding: str = 'utf-8'):
    mkdir_of_file(path)
    Path(path).write_text(text, encoding=encoding, errors='ignore')


def fast_min(x: T, y: T) -> T:
    return x if x < y else y


def fast_max(x: T, y: T) -> T:
    return x if x > y else y


def are_equal_arrs(a: array2D, b: array2D) -> bool:
    return a.shape == b.shape and (a == b).all()


def change_order(box: Box) -> Box:
    """
    >>> change_order((1, 2, 3, 4))
    (2, 1, 4, 3)
    """
    x1, y1, x2, y2 = box
    return y1, x1, y2, x2


def rectangles_have_intersections(rectangles: array2D) -> bool:
    """
    checks whether rectangles have intersections

    >>> _ = rectangles_have_intersections
    >>> _(np.array([(1, 2, 3, 4), (5, 6, 7, 8)]))
    False
    >>> _(np.array([(1, 2, 3, 4), (2, 3, 7, 8)]))
    True
    >>> _(np.array([[ 1,  2,  9,  5],
    ...   [ 4,  5,  7, 13],
    ...   [ 2,  9,  4, 11]]))
    True
    """

    if rectangles.shape[0] < 2:
        return False

    for i, (x1, y1, x2, y2) in enumerate(rectangles[:-1]):
        for _x1, _y1, _x2, _y2 in rectangles[i + 1:]:
            if (
                x1 > _x2 or y1 > _y2 or _x1 > x2 or _y1 > y2
            ):
                continue
            return True

    return False


#endregion


class RectTextViewer:
    """
    implements (array of coordinates rectangles) -> (str) conversion logic

    also supports vice versa conversion

    >>> rects = [(1, 2, 3, 4), (4, 6, 7, 8), (1, 6, 3, 9)]
    >>> vr = RectTextViewer(rects)
    >>> vrs = vr.to_string(show_order=True)
    >>> print(vrs) # doctest: +NORMALIZE_WHITESPACE
     1## 3###
     # # #  #
     ### ####
         2##
         # #
         # #
         ###
    >>> vr2 = RectTextViewer.from_string(vrs); assert vr == vr2
    """

    def __init__(self, rectangles: Union[arrayRectsInt, Iterable[BoxInt]]):

        if not isinstance(rectangles, np.ndarray):
            rectangles = np.array([r for r in rectangles])

        assert rectangles.shape[1] == 4, rectangles.shape
        assert (rectangles > 0).all()

        bad_rects_mask = (rectangles[:, 0] >= rectangles[:, 2]) | (rectangles[:, 1] >= rectangles[:, 3])
        if bad_rects_mask.any():
            raise ValueError(f"next rectangles are not valid: {rectangles[bad_rects_mask]}")

        self.rects = rectangles

    def __str__(self):
        return f'viewer of {self.rects.shape[0]} rectangles'

    def __eq__(self, other):
        return are_equal_arrs(self.rects, other.rects)

    @property
    def h_units(self) -> int:
        return self.rects[:, 2].max() - self.rects[:, 0].min() + 1

    @property
    def w_units(self) -> int:
        return self.rects[:, 3].max() - self.rects[:, 1].min() + 1

    @property
    def units(self):
        """
        count of units for the map

        >>> assert RectTextViewer([(1, 1, 2, 3)]).units == 3
        >>> assert RectTextViewer([(1, 2, 3, 4), (10, 11, 12, 13)]).units == 13
        """
        return self.rects[:, 2:].max() - self.rects[:, :2].min() + 1

    def to_array(self, show_order: bool = False) -> array2D:
        """
        >>> vr = RectTextViewer([(1, 1, 2, 3), (3, 4, 6, 7), (4, 1, 6, 2)])
        >>> vr.to_array()
        array([[-1, -1, -1, -2, -2, -2, -2],
               [-1, -1, -1, -2, -2, -2, -2],
               [-2, -2, -2, -1, -1, -1, -1],
               [-1, -1, -2, -1, -2, -2, -1],
               [-1, -1, -2, -1, -2, -2, -1],
               [-1, -1, -2, -1, -1, -1, -1]], dtype=int8)
        >>> vr.to_array(show_order=True)
        array([[ 1, -1, -1, -2, -2, -2, -2],
               [-1, -1, -1, -2, -2, -2, -2],
               [-2, -2, -2,  2, -1, -1, -1],
               [ 3, -1, -2, -1, -2, -2, -1],
               [-1, -1, -2, -1, -2, -2, -1],
               [-1, -1, -2, -1, -1, -1, -1]], dtype=int8)

        """
        xmax = self.rects[:, 2].max()
        ymax = self.rects[:, 3].max()

        arr = np.full((xmax, ymax), fill_value=_EMPTY_FILLER_INT, dtype=np.int8)

        for i, (x1, y1, x2, y2) in enumerate(self.rects, 1):
            x1 -= 1
            y1 -= 1
            arr[x1, y1: y2] = _BOUND_FILLER_INT
            arr[x2 - 1, y1: y2] = _BOUND_FILLER_INT
            arr[x1: x2, y1] = _BOUND_FILLER_INT
            arr[x1: x2, y2 - 1] = _BOUND_FILLER_INT

            if show_order:
                numbers = [int(s) for s in str(i)]
                arr[x1, y1: y1 + len(numbers)] = numbers

        return arr

    @staticmethod
    def from_array(arr: array2D):
        """
        >>> vr = RectTextViewer([(1, 1, 3, 3), (1, 5, 3, 7)])  # simple case
        >>> rr = vr.to_array(show_order=True); rr
        array([[ 1, -1, -1, -2,  2, -1, -1],
               [-1, -2, -1, -2, -1, -2, -1],
               [-1, -1, -1, -2, -1, -1, -1]], dtype=int8)
        >>> new = RectTextViewer.from_array(rr); assert vr == new

        >>> vr = RectTextViewer(  # hard case
        ...     np.array(
        ...         [
        ...             (1, 1, 2, 3),
        ...             (1, 4, 2, 8),
        ...             (3, 4, 6, 7),
        ...             (3, 1, 6, 2),
        ...             (3, 8, 7, 9)
        ...         ]
        ...     )
        ... )
        >>> rr = vr.to_array(show_order=True); vr.show() # doctest: +NORMALIZE_WHITESPACE
        1##2####
        ########
        4# 3###5#
        ## #  ###
        ## #  ###
        ## ######
               ##
        >>> new = RectTextViewer.from_array(rr); assert vr == new

        >>> vr = RectTextViewer(
        ...     [
        ...         (a, b, a + 1, b + 2)
        ...         for a in np.arange(1, 11, 2)
        ...         for b in np.arange(1, 41, 3)
        ...     ]
        ... )
        >>> vr.show() # doctest: +NORMALIZE_WHITESPACE
        1##2##3##4##5##6##7##8##9##10#11#12#13#14#
        ##########################################
        15#16#17#18#19#20#21#22#23#24#25#26#27#28#
        ##########################################
        29#30#31#32#33#34#35#36#37#38#39#40#41#42#
        ##########################################
        43#44#45#46#47#48#49#50#51#52#53#54#55#56#
        ##########################################
        57#58#59#60#61#62#63#64#65#66#67#68#69#70#
        ##########################################
        >>> rr = vr.to_array(show_order=True); new = RectTextViewer.from_array(rr); assert vr == new

        >>> vr = RectTextViewer(
        ...     [
        ...         ( 1,  1,  7,  4),
        ...         ( 3,  4,  6, 10),
        ...         ( 1,  7,  3,  9)
        ...     ]
        ... )
        >>> vr.show(show_order=True) # doctest: +NORMALIZE_WHITESPACE
        1###  3##
        #  #  # #
        #  2######
        #  #     #
        #  #     #
        #  #######
        ####
        >>> rr = vr.to_array(show_order=True); new = RectTextViewer.from_array(rr); assert vr == new
        """
        uniqs = np.unique(arr)

        if (uniqs == _EMPTY_FILLER_INT).all():
            raise ValueError(f"no rectangles found")

        unlabeled_mask = np.isin(uniqs, FILLERS_INT)

        if unlabeled_mask.all():
            raise ValueError(f"all rectangles are unlabeled")

        H, W = arr.shape
        arr_cp = arr.copy()

        rects = {}

        while True:
            for x, row in enumerate(arr_cp):
                digits_inds = np.nonzero(~np.isin(row, FILLERS_INT))[0]
                """indexes of digits"""
                if digits_inds.size != 0:
                    y = digits_inds[0]
                    break  # stop for loop cuz next x,y pair is found
            else:  # no digits found -- stop while loop
                break

            n = arr_cp[x, y]
            has_hole = arr_cp[x + 1, y + 1] == _EMPTY_FILLER_INT
            check_next_digits = True

            for _y in range(y + 1, W):  # seek for right bound
                v = row[_y]
                if (
                    v == _EMPTY_FILLER_INT or  # first empty
                    (v != _BOUND_FILLER_INT and not check_next_digits)  # first digit which is not for current number
                ):  # if not hole, the target if to find first empty
                    assert not has_hole
                    _y -= 1
                    break

                if check_next_digits:
                    if v != _BOUND_FILLER_INT:  # next digit
                        n = 10 * n + v
                    else:
                        check_next_digits = False  # stop checking on first mismatch

                if has_hole:  # if there is a hole, the target is to find the hole finish
                    if arr_cp[x + 1, _y] == _BOUND_FILLER_INT:  # it is right bound
                        break
            else:
                if has_hole:
                    raise Exception(f"rectangle starts on ({x}, {y}) is not matched (at right)")

            for _x in range(x + 1, H):  # seek for bottom bound
                if arr_cp[_x, y] != _BOUND_FILLER_INT:
                    assert not has_hole
                    _x -= 1
                    break
                if has_hole:
                    v = arr_cp[_x, y + 1]
                    if v == _BOUND_FILLER_INT:  # it is bottom bound
                        break
            else:
                if has_hole:
                    raise Exception(f"rectangle starts on ({x}, {y}) is not matched (at bottom)")

            assert n not in rects, f"{n} label repeats on the map"
            rects[n] = (x, y, _x, _y)
            # arr_cp[x: _x + 1, y: _y + 1] = _EMPTY_FILLER_INT
            arr_cp[x, y: y + len(str(n))] = _EMPTY_FILLER_INT  # remove only number label

        numbers = np.array(sorted(rects.keys()))
        all_numbers = np.arange(numbers[0], numbers[-1] + 1)
        if numbers.size != all_numbers.size:
            raise ValueError(
                f"next labels not found {[a for a in all_numbers if a not in numbers]}"
            )

        result = RectTextViewer(
            np.array([rects[n] for n in numbers]) + 1
        )

        diff_mask = result.to_array(show_order=True) != arr
        if diff_mask.any():
            r = arr.copy()
            r[~diff_mask] = _EMPTY_FILLER_INT
            raise ValueError(
                f"some mismatches found, possible bad structure or not all rectangles are labeled: {r}"
            )

        return result

    def to_string(self, show_order: bool = False):
        """
        >>> vr = RectTextViewer([(1, 1, 2, 3), (3, 4, 7, 8), (4, 1, 6, 2)])
        >>> print(vr.to_string(show_order=True))  # doctest: +NORMALIZE_WHITESPACE
        1##
        ###
           2####
        3# #   #
        ## #   #
        ## #   #
           #####
        """
        return '\n'.join(
            ''.join(
                (
                    EMPTY_FILLER_STR if n == _EMPTY_FILLER_INT else (
                        BOUND_FILLER_STR if n == _BOUND_FILLER_INT else (
                            str(n)
                        )
                    )
                )
                for n in line
            )
            for line in self.to_array(show_order=show_order)
        )

    @staticmethod
    def from_string(s: str):
        """
        >>> vr = RectTextViewer([(1, 1, 2, 3), (3, 4, 7, 8), (4, 1, 6, 2)])
        >>> st = vr.to_string(show_order=True)
        >>> assert vr == RectTextViewer.from_string(st)
        """
        return RectTextViewer.from_array(
            np.array(
                [
                    [
                        (
                            _EMPTY_FILLER_INT if v == EMPTY_FILLER_STR else (
                                _BOUND_FILLER_INT if v == BOUND_FILLER_STR else (
                                    int(v)
                                )
                            )
                        )
                        for v in line
                    ]
                    for line in s.strip('\n').splitlines()
                ]
            )
        )

    def show(self, show_order: bool = True):
        print(self.to_string(show_order=show_order))


class OrderedRectangles:
    """
    stores `(x1, y1, x2, y2)` numpy-ordered rectangles and provides an ability to
        1. show its text map which reflects the rectangles order and positions with different disretization level
        2. save/load the rectangles with/without the map to json file
        3. change the map manually to change the rectangles order easily

    How to create an object:
    >>> d = OrderedRectangles([(0.1, 0.2, 0.3, 0.4), (0.1, 0.6, 0.2, 1.1)])

    How to view the map with `units` discretization level:
    >>> units = 15
    >>> mp = d.get_order_map(units=units); print(mp) # doctest: +NORMALIZE_WHITESPACE
     1#### 2#######
     #   # #      #
     #   # ########
     #####

    Use `show_order_map` method to simplify this step:
    >>> d.show_order_map(units=units, show_order=False) # doctest: +NORMALIZE_WHITESPACE
     ##### ########
     #   # #      #
     #   # ########
     #####

    Let's try bigger example:
    >>> d = OrderedRectangles(
    ...     [
    ...         (0.1, 0.2, 0.3, 0.4), (0.1, 0.6, 0.2, 1.1),
    ...         (0.1, 1.2, 0.3, 1.4), (0.1, 1.5, 0.25, 2.3),
    ...         (0.35, 0.2, 0.6, 0.5), (0.4, 0.6, 0.6, 1.4)
    ...     ]
    ... )

    `units` <= 0 means to search the best value automatically
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

    I can swap 5th and 6th rectangles programmatically:
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

    The key feature is that I can change this order by changing the string map:
    >>> d3 = d[3]; d4 = d[4]
    >>> s = d.get_order_map(units=0).translate({ord('4'): '3', ord('3'): '4'})
    >>> d.load_order_map(s)
    >>> assert d[3] == d4 and d[4] == d3

    However it's not efficiet to change the rectangles order by changing the string map programmatically.
    Instead, it is supposed that u will save the map to some text file, change numbers manually and load the map from
        the file. `load_order_map` methods supports file path as an input too.

    Finally there is a way to save and load the rectangles as JSON:
    >>> tempdir = 'tmp'
    >>> jsdir = os.path.join(tempdir, 'separate-jsons'); mkdir(jsdir)
    >>> mapdir = os.path.join(tempdir, 'separate-maps'); mkdir(mapdir)
    >>> js_file = os.path.join(jsdir, 'rects.json')
    >>> map_file = os.path.join(mapdir, 'rects.txt')
    >>> d.to_json(path=js_file, save_map=map_file, units=0)
    >>> dct = read_json(js_file); print(str(dct).replace(os.sep, '/').replace('//', '/'))
    {'rects': [[0.1, 0.2, 0.3, 0.4], [0.1, 0.6, 0.2, 1.1], [0.1, 1.5, 0.25, 2.3], [0.1, 1.2, 0.3, 1.4], [0.4, 0.6, 0.6, 1.4], [0.35, 0.2, 0.6, 0.5]], 'map': '../separate-maps/rects.txt'}

    As u can see, tha path to maps file is saved as absolute or relative to the json file.
    U can change the order in the map manually and reload rectangles

    >>> write_text(map_file, read_text(map_file).translate({ord('1'): '2', ord('2'): '1'}))
    >>> d2 = OrderedRectangles.from_json(js_file)
    >>> assert d[1] == d2[2] and d[2] == d2[1]
    """

    def __init__(self, rectangles: Union[array2D, Iterable[BoxFloat]], is_numpy_ordered: bool = True):
        self.rects = rectangles if isinstance(rectangles, np.ndarray) else np.array([v for v in rectangles])
        if not is_numpy_ordered:
            self.rects[:, [0, 1]] = self.rects[:, [1, 0]]
            self.rects[:, [2, 3]] = self.rects[:, [3, 2]]

    def __eq__(self, other):
        return are_equal_arrs(self.rects, other.rects)

    def __getitem__(self, item):
        assert isinstance(item, int), item
        return tuple(self.rects[item - 1])

    def __setitem__(self, key, value):
        assert isinstance(key, int), key
        self.rects[key - 1] = value

    def as_list(self, numpy_ordered: bool = True) -> List[BoxFloat]:
        """
        >>> r = OrderedRectangles([(0.1, 0.2, 0.23, 1), (0.35, 0.45, 0.74, 0.8)])
        >>> assert r.as_list() == [(0.1, 0.2, 0.23, 1.0), (0.35, 0.45, 0.74, 0.8)]
        >>> r.as_list(numpy_ordered=False)
        [(0.2, 0.1, 1.0, 0.23), (0.45, 0.35, 0.8, 0.74)]
        >>> r2 = OrderedRectangles([(0.2, 0.1, 1.0, 0.23), (0.45, 0.35, 0.8, 0.74)], is_numpy_ordered=False)
        >>> assert r == r2
        """
        if numpy_ordered:
            return [tuple(row) for row in self.rects]
        return [change_order(row) for row in self.rects]

    def as_list_pil_ordered(self) -> List[BoxFloat]:
        return self.as_list(numpy_ordered=False)

    def get_best_units_count(
        self,
        minimum: int = UNITS_COUNT_SEARCH_MIN_VALUE,
        maximum: int = UNITS_COUNT_SEARCH_MAX_VALUE
    ) -> Tuple[int, arrayRectsInt]:
        """
        seeks for minimal units count to get the map without intersections
        Args:
            minimum:
            maximum:

        Returns:
            pair (units count, its map array object)

        >>> rects = np.array([(0, 0.1, 0.6, 0.3), (0.3, 0.4, 0.5, 1), (0.1, 0.7, 0.2, 0.8)])
        >>> d = OrderedRectangles(rects)
        >>> u, m = d.get_best_units_count()
        >>> u
        19
        >>> RectTextViewer(m).show() # doctest: +NORMALIZE_WHITESPACE
         1#####
         #    #     3###
         #    #     #  #
         #    #     #  #
         #    #     ####
         #    #2###########
         #    ##          #
         #    ##          #
         #    ##          #
         #    #############
         #    #
         ######
        >>> d.show_order_map(units=u - 1)  # has intersections # doctest: +NORMALIZE_WHITESPACE
         1#####
         #    #    3###
         #    #    #  #
         #    #    #  #
         #    #    ####
         #    2###########
         #    #          #
         #    #          #
         #    #          #
         #    ############
         #    #
         ######
        >>> rects = np.array([(0, 0.1, 0.6, 0.3), (0.3, 0.4, 0.5, 1), (0.1, 0.63, 0.2, 5)])
        >>> d = OrderedRectangles(rects)
        >>> u, m = d.get_best_units_count()
        >>> u
        89
        >>> d.show_order_map(units=u)  # doctest: +NORMALIZE_WHITESPACE
         1#####
         #    #    3#############################################################################
         #    #    #                                                                            #
         #    #    #                                                                            #
         #    #    ##############################################################################
         #    #2###########
         #    ##          #
         #    ##          #
         #    ##          #
         #    #############
         #    #
         ######
        """

        assert not rectangles_have_intersections(self.rects)

        r = self.get_discretized_array(minimum)
        if not rectangles_have_intersections(r):
            return minimum, r

        res = self.get_discretized_array(maximum)
        if rectangles_have_intersections(res):
            raise Exception(f"cannot find optimal unit cuz max unit size {maximum} failed, try to increase")

        a = minimum
        b = maximum
        while b - a > 1:
            c = (b + a) // 2
            r = self.get_discretized_array(c)
            if rectangles_have_intersections(r):
                a = c
            else:
                b = c
                res = r

        return b, res

    def get_discretized_array(self, units: int = 10) -> arrayRectsInt:
        """
        Args:
            units: discretization level, values <=0 mean to search mininal valid automatically

        >>> r = OrderedRectangles([(1, 2, 3, 4), (5, 6, 7, 8)])
        >>> r.get_discretized_array(8)
        array([[1, 2, 3, 4],
               [5, 6, 7, 8]])
        >>> r.get_discretized_array(4)
        array([[1, 1, 2, 3],
               [2, 3, 4, 4]])

        >>> r = OrderedRectangles([(0.1, 0.04, 0.3, 0.22), (0.87, 0.6, 1.5, 0.9)])
        >>> r.get_discretized_array(12)
        array([[ 1,  1,  3,  3],
               [ 7,  5, 12,  8]])
        >>> r.get_discretized_array(25)
        array([[ 1,  1,  6,  4],
               [14, 10, 25, 16]])
        >>> r.get_discretized_array(0)  # most compact case
        array([[1, 1, 2, 2],
               [3, 2, 5, 4]])
        """

        if units <= 0:
            return self.get_best_units_count(
                minimum=UNITS_COUNT_SEARCH_MIN_VALUE if units == 0 else -units,
                maximum=UNITS_COUNT_SEARCH_MAX_VALUE
            )[1]

        # x1, y1, x2, y2 = self.rects.T.copy()

        # xmin = x1.min()
        # xmax = x2.max()
        # xcoef = h_units / (xmax - xmin)
        # """coef to convert initial range to [1; h_units] range"""
        #
        # x1 = np.floor((x1 - xmin + 1) * xcoef)
        # x2 = np.ceil((x2 - xmin + 1) * xcoef)
        #
        # ymin = y1.min()
        # ymax = y2.max()
        # ycoef = w_units / (ymax - ymin)
        #
        # y1 = np.floor((y1 - ymin + 1) * ycoef)
        # y2 = np.ceil((y2 - ymin + 1) * ycoef)

        # return np.array((x1, y1, x2, y2)).T.astype(int)

        mn = self.rects[:, :2].min()
        mx = self.rects[:, 2:].max()

        arr = (self.rects - mn) * ((units - 1) / (mx - mn))
        arr[:, :2] = np.floor(arr[:, :2])
        arr[:, 2:] = np.ceil(arr[:, 2:])
        return (
            np.minimum(units, arr.astype(int) + 1)
        )  # minimum is necessary due to precision errors sometimes

    def get_order_map(self, units: int = 10, show_order: bool = True):
        arr = self.get_discretized_array(units=units)
        return RectTextViewer(arr).to_string(show_order=show_order)

    def show_order_map(self, **get_order_map_kwargs):
        """
        >>> r = OrderedRectangles([(0.1, 0.2, 0.23, 1), (0.35, 0.45, 0.74, 0.8)])
        >>> r.show_order_map(units=12)  # doctest: +NORMALIZE_WHITESPACE
         1##########
         #         #
         ###########
            2#####
            #    #
            #    #
            #    #
            #    #
            ######
        """
        print(self.get_order_map(**get_order_map_kwargs))

    def save_order_map(self, path: PathLike, **get_order_map_kwargs):
        """saves text order map to file"""
        write_text(
            path, self.get_order_map(**get_order_map_kwargs)
        )

    def load_order_map(self, order_map: Union[RectTextViewer, str, PathLike]):
        """
        parses order map to change the rectangles order in the current object
        Args:
            order_map: order map view, its string or the file with its string

        >>> rects = np.array([(0, 0.1, 0.6, 0.3), (0.3, 0.4, 0.5, 1), (0.1, 0.7, 0.2, 0.8)])  # some rects
        >>> def check(units: int):
        ...     r = OrderedRectangles(rects)  # make object from rects
        ...     mp = r.get_order_map(units=units)  # make the map from the object with input discretization level
        ...     vr = RectTextViewer.from_string(mp); vr.rects = vr.rects[::-1]  # force change rectangles order (reverse)
        ...     r.load_order_map(vr)  # load order to the object
        ...     assert are_equal_arrs(r.rects, rects[::-1])  # check whether the order is reversed
        >>> check(9)
        >>> check(10)
        >>> check(20)
        >>> check(120)

        >>> rects = np.vstack((rects, rects + 2))
        >>> r1 = OrderedRectangles(rects)
        >>> random_indexes = np.random.permutation(rects.shape[0])
        >>> r2 = OrderedRectangles(rects[random_indexes])
        >>> r1.load_order_map(r2.get_order_map(units=50))  # transfer order through the map
        >>> assert are_equal_arrs(r1.rects, rects[random_indexes])  # check the transfer is successful

        """
        if not isinstance(order_map, RectTextViewer):  # string or path
            if any(s.isalpha() for s in str(order_map)):  # path
                order_map = read_text(order_map)
            # here its a viewer string content
            order_map = RectTextViewer.from_string(order_map)

        # here it is an viewer object

        current_int_rects = self.get_discretized_array(units=order_map.units)

        assert current_int_rects.shape == order_map.rects.shape

        current_int_to_index: Dict[BoxInt, int] = {
            tuple(row): i for i, row in enumerate(current_int_rects.tolist())
        }

        target_int_rects: List[BoxInt] = [
            tuple(row) for row in order_map.rects.tolist()
        ]
        """int rects from the map in the target order"""

        ordered_rects = np.array(
            [
                self.rects[current_int_to_index[r]] for r in target_int_rects
            ]
        )
        self.rects = ordered_rects

    def to_json(self, path: PathLike, save_map: Union[bool, PathLike] = True, **get_order_map_kwargs):
        """
        saves the rectangles to json
        Args:
            path: json path
            save_map: whether to save the map too, path means to save in the other file instead of json directly
            **get_order_map_kwargs:

        Returns:

        Notes:
            loading object with the map is very helpful
            cuz u can view the current rectangles order and change it manually and then load the objects with new order
        """
        result = {
            'rects': [tuple(row) for row in self.rects.tolist()],
            'map': None
        }

        if save_map:
            mp = self.get_order_map(**get_order_map_kwargs)
            if isinstance(save_map, bool):  # save the string inplace
                result['map'] = mp
            else:
                write_text(save_map, mp)
                result['map'] = os.path.relpath(Path(save_map), Path(path).parent)

        write_json(path, result)

    @staticmethod
    def from_json(path: PathLike):
        """
        >>> import tempfile
        >>> rects = np.array([(0, 0.1, 0.6, 0.3), (0.3, 0.4, 0.5, 1), (0.1, 0.7, 0.2, 0.8)])
        >>> obj = OrderedRectangles(rects)
        >>> file = tempfile.mktemp(suffix='.json')
        >>> obj.to_json(file, save_map=False)
        >>> assert obj == OrderedRectangles.from_json(file)
        >>> obj.to_json(file, save_map=True, units=13)
        >>> assert obj == OrderedRectangles.from_json(file)
        """
        data = read_json(path)
        result = OrderedRectangles(data['rects'])

        mp = data.get('map')
        if mp:
            if any(s.isalpha() for s in mp):  # path
                if not os.path.isabs(mp):
                    mp = Path(path).parent / mp
            result.load_order_map(mp)

        return result


def main():
    v = RectTextViewer(
        np.array(
            [
                (1, 1, 2, 3),
                (1, 4, 2, 8),
                (3, 4, 6, 7),
                (3, 1, 6, 2),
                (3, 8, 7, 9)
            ]
        )
    )

    r = v.to_array(show_order=True)
    RectTextViewer.from_array(r)

    print()


if __name__ == '__main__':
    main()
