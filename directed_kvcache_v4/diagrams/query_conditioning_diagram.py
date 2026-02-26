from manim import *
import numpy as np


# Shared constants (from attention_progression.py)
SQUARE_SIZE = 0.65
SPACING = 0.95
DARK_TEXT = "#1a1a1a"
MID_TEXT = "#555555"

# Color palette
DOC_LIGHT = "#BFDBFE"          # very light blue
DOC_MID = "#60A5FA"            # medium blue (enriched)
DOC_BORDER_LIGHT = "#93C5FD"
DOC_BORDER_MID = "#3B82F6"

QUERY_LIGHT = "#FDE68A"        # very light yellow
QUERY_MID = "#FBBF24"          # medium yellow
QUERY_BORDER_LIGHT = "#FCD34D"
QUERY_BORDER_MID = "#D97706"

ARROW_BLUE = "#3B82F6"
ARROW_YELLOW = "#D97706"

N_DOC = 6
N_QUERY = 4


def make_tokens(positions, labels, fill, border, opacity=0.85):
    """Create rounded-rect tokens with labels."""
    squares = VGroup()
    texts = VGroup()
    for pos, label in zip(positions, labels):
        sq = RoundedRectangle(
            corner_radius=0.08,
            width=SQUARE_SIZE, height=SQUARE_SIZE,
            fill_color=fill, fill_opacity=opacity,
            stroke_color=border, stroke_width=2.5,
        ).move_to(pos)
        squares.add(sq)
        txt = Text(label, font_size=18, weight=BOLD, color=DARK_TEXT)
        txt.move_to(pos)
        texts.add(txt)
    return squares, texts


def make_brace(positions, label_text, color, direction=DOWN):
    """Create a brace below a group of tokens."""
    left = positions[0][0] - SQUARE_SIZE / 2
    right = positions[-1][0] + SQUARE_SIZE / 2
    y = positions[0][1] - SQUARE_SIZE / 2 - 0.12
    brace = BraceBetweenPoints(
        [left, y, 0], [right, y, 0],
        direction=direction, color=color,
    )
    label = Text(label_text, font_size=22, color=color, weight=BOLD)
    label.next_to(brace, DOWN, buff=0.1)
    return brace, label


def make_column_box(center, width, height, fill, border, opacity=0.12, corner_r=0.2):
    """Create a large rounded rectangle as a column background."""
    return RoundedRectangle(
        corner_radius=corner_r,
        width=width, height=height,
        fill_color=fill, fill_opacity=opacity,
        stroke_color=border, stroke_width=2.0,
    ).move_to(center)


class QueryConditioningDiagram(Scene):
    """
    Left column:  Query (yellow)
    Right column: Document (blue, top) and Query-conditioned Document (blue, bottom)
    Arrow shows the conditioning flow from query to document.
    """

    def construct(self):
        self.camera.background_color = WHITE

        # Layout parameters
        left_x = -3.5      # query column center
        right_x = 2.0      # document column center
        top_y = 2.0         # top row vertical center
        bottom_y = -1.2     # bottom row vertical center
        col_width = 3.2     # column background width

        # --- Title ---
        title = Text("Query-Conditioned Document Encoding",
                      font_size=34, weight=BOLD, color=DARK_TEXT)
        title.to_edge(UP, buff=0.35)
        self.add(title)

        # === LEFT COLUMN: Query ===
        # Column background
        q_col_bg = make_column_box(
            center=np.array([left_x, (top_y + top_y - (N_QUERY - 1) * SPACING) / 2, 0]),
            width=col_width, height=N_QUERY * SPACING + 1.8,
            fill=QUERY_LIGHT, border=QUERY_BORDER_LIGHT, opacity=0.10,
        )
        self.add(q_col_bg)

        # Column header
        q_header = Text("Query", font_size=26, weight=BOLD, color=ARROW_YELLOW)
        q_header.move_to([left_x, top_y + 0.9, 0])
        self.add(q_header)

        # Query tokens (vertical stack)
        q_labels = [f"q{i+1}" for i in range(N_QUERY)]
        q_positions = [np.array([left_x, top_y - i * SPACING, 0]) for i in range(N_QUERY)]
        q_squares, q_texts = make_tokens(
            q_positions, q_labels, QUERY_LIGHT, QUERY_BORDER_MID)
        self.add(q_squares, q_texts)

        # Ellipsis below query tokens
        q_dots = Text("...", font_size=24, color=MID_TEXT)
        q_dots.move_to([left_x, q_positions[-1][1] - 0.7, 0])
        self.add(q_dots)

        # === RIGHT COLUMN: Document (top) + Conditioned Document (bottom) ===
        # Column background
        d_col_bg = make_column_box(
            center=np.array([right_x, (top_y + bottom_y) / 2, 0]),
            width=col_width + 1.4, height=top_y - bottom_y + 2.8,
            fill=DOC_LIGHT, border=DOC_BORDER_LIGHT, opacity=0.10,
        )
        self.add(d_col_bg)

        # --- Top section: plain Document ---
        d_header = Text("Document", font_size=26, weight=BOLD, color=ARROW_BLUE)
        d_header.move_to([right_x, top_y + 0.9, 0])
        self.add(d_header)

        # Document tokens (horizontal row)
        doc_start_x = right_x - (N_DOC - 1) * SPACING / 2
        d_labels = [f"d{i+1}" for i in range(N_DOC)]
        d_positions = [np.array([doc_start_x + i * SPACING, top_y, 0]) for i in range(N_DOC)]
        d_squares, d_texts = make_tokens(
            d_positions, d_labels, DOC_LIGHT, DOC_BORDER_LIGHT)
        self.add(d_squares, d_texts)

        d_brace, d_blabel = make_brace(d_positions, "Original KV Cache", DOC_BORDER_LIGHT)
        self.add(d_brace, d_blabel)

        # --- Bottom section: Query-conditioned Document ---
        cd_header = Text("Query-Conditioned Document", font_size=22,
                         weight=BOLD, color=DOC_BORDER_MID)
        cd_header.move_to([right_x, bottom_y + 1.2, 0])
        self.add(cd_header)

        cd_positions = [np.array([doc_start_x + i * SPACING, bottom_y, 0]) for i in range(N_DOC)]
        cd_labels = [f"d{i+1}'" for i in range(N_DOC)]
        cd_squares, cd_texts = make_tokens(
            cd_positions, cd_labels, DOC_MID, DOC_BORDER_MID)
        self.add(cd_squares, cd_texts)

        cd_brace, cd_blabel = make_brace(cd_positions, "Enriched KV Cache", DOC_BORDER_MID)
        self.add(cd_brace, cd_blabel)

        # === ARROWS ===
        # Downward arrow from plain doc brace to conditioned header
        arrow_mid_y = (top_y + bottom_y) / 2 + 0.15
        down_arrow = Arrow(
            start=np.array([right_x + 2.0, top_y - SQUARE_SIZE / 2 - 0.15, 0]),
            end=np.array([right_x + 2.0, bottom_y + SQUARE_SIZE / 2 + 0.15, 0]),
            color=MID_TEXT, stroke_width=2.5, buff=0.05,
            max_tip_length_to_length_ratio=0.12,
        )
        self.add(down_arrow)

        # Curved arrow from query column to the midpoint between rows
        arrow_start = np.array([left_x + SQUARE_SIZE / 2 + 0.2,
                                (top_y + q_positions[-1][1]) / 2, 0])
        arrow_end = np.array([d_positions[0][0] - SQUARE_SIZE / 2 - 0.15,
                              arrow_mid_y, 0])

        # Curved path via bezier
        mid_x = (arrow_start[0] + arrow_end[0]) / 2
        mid_y = arrow_start[1] + 0.6
        n_pts = 40
        ts = np.linspace(0, 1, n_pts)
        curve_pts = []
        for t in ts:
            px = (1-t)**2 * arrow_start[0] + 2*(1-t)*t * mid_x + t**2 * arrow_end[0]
            py = (1-t)**2 * arrow_start[1] + 2*(1-t)*t * mid_y + t**2 * arrow_end[1]
            curve_pts.append([px, py, 0])

        curve = VMobject(color=ARROW_YELLOW, stroke_width=3.0, stroke_opacity=0.8)
        curve.set_points_smoothly([np.array(p) for p in curve_pts])
        self.add(curve)

        # Arrowhead at end of curve
        last = np.array(curve_pts[-1])
        prev = np.array(curve_pts[-3])
        direction = last - prev
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0], 0])
        tip_size = 0.12
        tip = Polygon(
            last,
            last - tip_size * direction + tip_size * 0.5 * perp,
            last - tip_size * direction - tip_size * 0.5 * perp,
            fill_color=ARROW_YELLOW, fill_opacity=0.8,
            stroke_width=0,
        )
        self.add(tip)

        # Label on the curve
        cond_label = Text("condition", font_size=18, color=ARROW_YELLOW,
                          weight=BOLD, slant=ITALIC)
        cond_label.move_to([mid_x - 0.3, mid_y + 0.2, 0])
        self.add(cond_label)

        # Small annotation below conditioned row
        note = Text(
            "Document KV states enriched by query attention",
            font_size=14, color=MID_TEXT,
        )
        note.next_to(cd_blabel, DOWN, buff=0.25)
        self.add(note)
