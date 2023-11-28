
from typing import Tuple, Dict, Union, Optional, Sequence


from reportlab.lib.colors import Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from .main import BoxFloat, PathLike, mkdir_of_file


# region UTILS

# region colors

def _as_Color(rgb: Tuple[int, int, int], alpha: float = 1):
    return Color(rgb[0], rgb[1], rgb[2], alpha=alpha)


default_pdf_colors = {
    'blocks_rgb': [0, 255, 0],
    'blocks_alpha': 0.3,
    'blocks_stroke_rgb': [10, 255, 0],
    'blocks_stroke_alpha': 1.0,
    'blocks_stroke_width': 1,
    'blocks_text_rgb': [255, 0, 0],
    'blocks_text_alpha': 1.0,
}


class DrawingData:
    """
    collection of colors and other info for pdf drawing
    """

    def __init__(
        self,
        pdf_colors: Optional[Dict[str, Union[float, Tuple[int, int, int]]]] = None,
    ):
        # прочесть параметры, если надо
        if pdf_colors is None:
            pdf_colors = default_pdf_colors.copy()
        else:  # dict
            _defaults = default_pdf_colors.copy()
            _defaults.update(pdf_colors)
            pdf_colors = _defaults

        def name_to_color(name: str, mult: float = 1):
            val = pdf_colors.get(f"{name}_rgb")
            if val is None:
                return None

            rgb = tuple(val)
            alp = pdf_colors.get(f"{name}_alpha", 1)
            return _as_Color(rgb, alp * mult)

        def name_to_width(name: str):
            return pdf_colors.get(f"{name}_stroke_width", 0.1)


        self.COLOR_BLOCK = name_to_color('blocks')
        self.COLOR_BLOCK_STROKE = name_to_color('blocks_stroke')
        self.BLOCK_STROKE = name_to_width('blocks')
        self.COLOR_BLOCK_TEXT = name_to_color('blocks_text')


# endregion

# region boxes

def transform_bbox(bbox: BoxFloat, y_bottom: float):
    """
    multiplies PIL bbox by 1000 and transforms by `y` to draw on pdf

    >>> transform_bbox((200, 300, 400, 500), y_bottom=800)
    (200.0, 500.0, 400.0, 300.0)
    """
    x1, y1, x2, y2 = bbox
    y1, y2 = y_bottom - y1,  y_bottom - y2
    return x1, y1, x2, y2

# endregion

# endregion


# @profile
def save_rectangles_to_pdf(
    pages_sizes: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    pages_rectangles: Sequence[Sequence[BoxFloat]],
    path_to_save: PathLike,
    pdf_colors: Optional[Dict[str, Union[float, Tuple[int, int, int]]]] = None,
    font_default: str = 'Helvetica',
):

    # region INITIALS

    page_count = len(pages_rectangles)
    assert page_count, 'no pages'
    if isinstance(pages_sizes, tuple) and isinstance(pages_sizes[0], (int, float)):  # one size for all pages
        pages_sizes = [pages_sizes] * page_count
    else:
        assert len(pages_sizes) == page_count, (len(pages_sizes), len(pages_rectangles))

    draw = DrawingData(
        pdf_colors=pdf_colors,
    )

    mkdir_of_file(path_to_save)

    # endregion

    p = None
    """actual canvas object"""

    for page_number in range(page_count):  # drawing process for each page

        # region INITIALS
        pagesize = pages_sizes[page_number]
        blocks_on_page = pages_rectangles[page_number]

        y_bottom = pagesize[-1]

        if page_number == 0:  # create canvas on first page iteration
            p = Canvas(path_to_save, pagesize=pagesize)
        else:
            p.showPage()  # старую страницу нарисовать
            p.setPageSize(pagesize)  # сделать новую с таким размером

        def draw_rect(box: BoxFloat):
            _x1, _y1, _x2, _y2 = transform_bbox(box, y_bottom)
            p.rect(_x1, _y1, _x2 - _x1, _y2 - _y1, fill=1)

        # endregion


        # region draw blocks / lines / tokens rectangles (on flags to show their borders and order)
        #
        for fill_color, stroke_color, stroke_width, seq in (
            (draw.COLOR_BLOCK, draw.COLOR_BLOCK_STROKE, draw.BLOCK_STROKE, blocks_on_page),
        ):
            if not blocks_on_page:
                continue

            p.setLineWidth(stroke_width)
            p.setStrokeColor(stroke_color)
            p.setFillColor(fill_color)
            for b in seq:
                draw_rect(b)
        # endregion


        # region draw numbers of blocks (if foreground of other drawings)
        for color, seq, position in (
            (draw.COLOR_BLOCK_TEXT, blocks_on_page, 'center'),
        ):
            if not blocks_on_page:
                continue

            p.setFillColor(color)
            for i, b in enumerate(seq):
                x1, y1, x2, y2 = transform_bbox(b, y_bottom)

                face = pdfmetrics.getFont(font_default).face
                fontsize = 0.5 * abs(y2 - y1) / (face.ascent + face.descent) * 1000
                text = str(i + 1)
                width = abs(p.stringWidth(text, font_default, fontsize))

                font_scale = (x2 - x1) / width
                if font_scale < 1:
                    fontsize *= font_scale
                    width = abs(p.stringWidth(text, font_default, fontsize))

                p.setFont(font_default, fontsize)

                if position == 'left':
                    x = x1
                elif position == 'right':
                    x = x2 - width
                else:
                    x = x1 + (x2 - x1 - width) / 2

                p.drawString(x, y2, text)
        # endregion

    p.showPage()
    p.save()

