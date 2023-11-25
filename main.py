
from typing import Union, Iterable, Tuple
from typing_extensions import TypeAlias

import numpy as np

BoxFloat: TypeAlias = Tuple[float, float, float, float]
BoxInt: TypeAlias = Tuple[int, int, int, int]

array2D: TypeAlias = np.ndarray
arrayRects: TypeAlias = array2D
"""2D array of rectangles with columns (x1, y1, x2, y2) where x is rows dim"""
arrayRectsInt: TypeAlias = arrayRects
"""
2D array with integer rectangles coordinates in form (x1, y1, x2, y2)

where (1, 2, 3, 4) means to fill [1:3, 2:4] including bounds, one-based
"""

_EMPTY_FILLER_INT = -2
_BOUND_FILLER_INT = -1

FILLERS_INT = (_BOUND_FILLER_INT, _EMPTY_FILLER_INT)

_EMPTY_FILLER_STR = ' '
_BOUND_FILLER_STR = '#'


class RectTextViewer:

    def __init__(self, rectangles: arrayRectsInt):

        assert rectangles.shape[1] == 4, rectangles.shape
        assert (rectangles > 0).all()

        bad_rects_mask = (rectangles[:, 0] >= rectangles[:, 2]) | (rectangles[:, 1] >= rectangles[:, 3])
        if bad_rects_mask.any():
            raise ValueError(f"next rectangles are not valid: {rectangles[bad_rects_mask]}")

        self.rects = rectangles

    def __str__(self):
        return f'viewer of {self.rects.shape[0]} rectangles'

    def __eq__(self, other):
        return self.rects.shape == other.rects.shape and (self.rects == other.rects).all()

    @property
    def h_units(self) -> int:
        return self.rects[:, 2].max() - self.rects[:, 0].min() + 1

    @property
    def w_units(self) -> int:
        return self.rects[:, 3].max() - self.rects[:, 1].min() + 1

    @property
    def units(self):
        return max(self.h_units, self.w_units)

    def to_array(self, show_order: bool = False) -> array2D:
        """
        >>> vr = RectTextViewer(np.array([(1, 1, 2, 3), (3, 4, 6, 7), (4, 1, 6, 2)]))
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
        >>> vr = RectTextViewer(np.array([(1, 1, 3, 3), (1, 5, 3, 7)]))  # simple case
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
        ...     np.array([
        ...         (a, b, a + 1, b + 2)
        ...         for a in np.arange(1, 11, 2)
        ...         for b in np.arange(1, 41, 3)
        ...     ])
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

            rects[n] = (x, y, _x, _y)
            arr_cp[x: _x + 1, y: _y + 1] = _EMPTY_FILLER_INT

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
        >>> vr = RectTextViewer(np.array([(1, 1, 2, 3), (3, 4, 7, 8), (4, 1, 6, 2)]))
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
                    _EMPTY_FILLER_STR if n == _EMPTY_FILLER_INT else (
                        _BOUND_FILLER_STR if n == _BOUND_FILLER_INT else (
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
        >>> vr = RectTextViewer(np.array([(1, 1, 2, 3), (3, 4, 7, 8), (4, 1, 6, 2)]))
        >>> st = vr.to_string(show_order=True)
        >>> assert vr == RectTextViewer.from_string(st)
        """
        return RectTextViewer.from_array(
            np.array(
                [
                    [
                        (
                            _EMPTY_FILLER_INT if v == _EMPTY_FILLER_STR else (
                                _BOUND_FILLER_INT if v == _BOUND_FILLER_STR else (
                                    int(v)
                                )
                            )
                        )
                        for v in line
                    ]
                    for line in s.strip().splitlines()
                ]
            )
        )

    def show(self, show_order: bool = True):
        print(self.to_string(show_order=show_order))


class OrderedRectangles:
    def __init__(self, rectangles: Union[array2D, Iterable[BoxFloat]]):
        self.rects = rectangles if isinstance(rectangles, np.ndarray) else np.array([v for v in rectangles])

    def get_discretized_array(self, units: int = 10) -> arrayRectsInt:
        """
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
               [14, 10, 26, 16]])
        """
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
        return arr.astype(int) + 1

    def show(self, units: int = 10, show_order: bool = True):
        """
        >>> r = OrderedRectangles([(0.1, 0.2, 0.23, 1), (0.35, 0.45, 0.74, 0.8)])
        >>> r.show(units=12)  # doctest: +NORMALIZE_WHITESPACE
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
        arr = self.get_discretized_array(units=units)
        RectTextViewer(arr).show(show_order=show_order)


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
