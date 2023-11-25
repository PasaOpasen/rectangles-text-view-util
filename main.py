
from typing_extensions import TypeAlias

import numpy as np


array2D: TypeAlias = np.ndarray
arrayRects: TypeAlias = array2D
"""2D array of rectangles with columns (x1, y1, x2, y2) where x is rows dim"""
arrayRectsInt: TypeAlias = arrayRects
"""
2D array with integer rectangles coordinates in form (x1, y1, x2, y2)

where (1, 2, 3, 4) means to fill [1:3, 2:4] including bounds, one-based
"""

_EMPTY_FILLER_INT = 0
_BOUND_FILLER_INT = -1

_EMPTY_FILLER_STR = ' '
_BOUND_FILLER_STR = '#'


class RectTextViewer:

    def __init__(self, rectangles: arrayRectsInt):
        self.rects = rectangles

    def to_array(self, show_order: bool = False) -> array2D:
        """
        >>> vr = RectTextViewer(np.array([(1, 1, 2, 3), (3, 4, 6, 7), (4, 1, 6, 2)]))
        >>> vr.to_array()
        array([[-1, -1, -1,  0,  0,  0,  0],
               [-1, -1, -1,  0,  0,  0,  0],
               [ 0,  0,  0, -1, -1, -1, -1],
               [-1, -1,  0, -1,  0,  0, -1],
               [-1, -1,  0, -1,  0,  0, -1],
               [-1, -1,  0, -1, -1, -1, -1]], dtype=int8)
        >>> vr.to_array(show_order=True)
        array([[ 1, -1, -1,  0,  0,  0,  0],
               [-1, -1, -1,  0,  0,  0,  0],
               [ 0,  0,  0,  2, -1, -1, -1],
               [ 3, -1,  0, -1,  0,  0, -1],
               [-1, -1,  0, -1,  0,  0, -1],
               [-1, -1,  0, -1, -1, -1, -1]], dtype=int8)

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

    def show(self, show_order: bool = True):
        print(self.to_string(show_order=show_order))


def main():
    v = RectTextViewer(
        np.array(
            [
                (1, 1, 2, 3),
                (3, 4, 6, 7)
            ]
        )
    )

    v.to_array()

    print()


if __name__ == '__main__':
    main()
