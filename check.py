
import numpy as np
from ordered_rectangles.main import RectTextViewer


def main1():
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


def main2():
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
    main1()

