from manim import *
import numpy as np


# Shared constants
SQUARE_SIZE = 0.65
SPACING = 0.95
DARK_TEXT = "#1a1a1a"
MID_TEXT = "#555555"

# Color palette
DOC_LIGHT = "#BFDBFE"      # very light blue
DOC_MID = "#60A5FA"         # medium blue (enriched)
DOC_BORDER_LIGHT = "#93C5FD"
DOC_BORDER_MID = "#3B82F6"

QUERY_LIGHT = "#FDE68A"     # very light yellow
QUERY_MID = "#FBBF24"       # medium yellow (enriched)
QUERY_BORDER_LIGHT = "#FCD34D"
QUERY_BORDER_MID = "#D97706"

SURROGATE_LIGHT = "#FED7AA"   # light orange/peach
SURROGATE_BORDER = "#EA580C"  # burnt orange border

ARROW_BLUE = "#3B82F6"
ARROW_YELLOW = "#D97706"

N_DOC = 5
N_QUERY = 4
N_SURROGATE = 4


def make_tokens(positions, labels, fill, border, opacity=0.85):
    """Create rounded-rect tokens with labels."""
    squares = VGroup()
    texts = VGroup()
    for i, (pos, label) in enumerate(zip(positions, labels)):
        sq = RoundedRectangle(
            corner_radius=0.08,
            width=SQUARE_SIZE, height=SQUARE_SIZE,
            fill_color=fill, fill_opacity=opacity,
            stroke_color=border, stroke_width=2.5,
        ).move_to(pos)
        squares.add(sq)
        txt = Text(label, font_size=20, weight=BOLD, color=DARK_TEXT)
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


def make_arc(x_start, x_end, y_top, arc_h, color, opacity, stroke_w):
    """Create a bezier arc with arrowhead."""
    x_mid = (x_start + x_end) / 2
    y_mid = y_top + arc_h

    n_points = 30
    ts = np.linspace(0, 1, n_points)
    points = []
    for t in ts:
        px = (1 - t)**2 * x_start + 2*(1 - t)*t * x_mid + t**2 * x_end
        py = (1 - t)**2 * y_top + 2*(1 - t)*t * y_mid + t**2 * y_top
        points.append([px, py, 0])

    curve = VMobject(color=color, stroke_width=stroke_w, stroke_opacity=opacity)
    curve.set_points_smoothly([np.array(p) for p in points])

    # Arrowhead
    end_pt = np.array(points[-1])
    prev_pt = np.array(points[-3])
    d = end_pt - prev_pt
    d = d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0], 0])
    ts2 = 0.09
    tip = Polygon(
        end_pt,
        end_pt - d * ts2 * 1.5 + perp * ts2 * 0.6,
        end_pt - d * ts2 * 1.5 - perp * ts2 * 0.6,
        fill_color=color, fill_opacity=opacity, stroke_width=0,
    )
    return VGroup(curve, tip)


def make_self_loop(pos, color, opacity=0.6):
    """Self-attention loop: arcs from top-right to top-left of same token."""
    cx = pos[0]
    top_y = pos[1] + SQUARE_SIZE / 2 + 0.06
    x_s = cx + SQUARE_SIZE * 0.22
    x_e = cx - SQUARE_SIZE * 0.22
    arc_h = 0.35
    x_mid = cx
    y_mid = top_y + arc_h

    n_pts = 20
    ts = np.linspace(0, 1, n_pts)
    pts = []
    for t in ts:
        px = (1 - t)**2 * x_s + 2*(1 - t)*t * x_mid + t**2 * x_e
        py = (1 - t)**2 * top_y + 2*(1 - t)*t * y_mid + t**2 * top_y
        pts.append([px, py, 0])

    curve = VMobject(color=color, stroke_width=1.8, stroke_opacity=opacity)
    curve.set_points_smoothly([np.array(p) for p in pts])

    end_pt = np.array(pts[-1])
    prev_pt = np.array(pts[-2])
    d = end_pt - prev_pt
    d = d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0], 0])
    ts2 = 0.065
    tip = Polygon(
        end_pt,
        end_pt - d * ts2 * 1.5 + perp * ts2 * 0.6,
        end_pt - d * ts2 * 1.5 - perp * ts2 * 0.6,
        fill_color=color, fill_opacity=opacity, stroke_width=0,
    )
    return VGroup(curve, tip)


def make_causal_arcs(positions, n_tokens, color, start_idx=0):
    """Generate all causal attention arcs for tokens at given positions."""
    arcs = VGroup()
    ARC_BASE = 0.5
    ARC_STEP = 0.28

    for i in range(start_idx, start_idx + n_tokens):
        for j in range(start_idx, i):
            dist = i - j
            arc_h = ARC_BASE + ARC_STEP * (dist - 1)
            y_top = positions[i][1] + SQUARE_SIZE / 2 + 0.06
            opacity = max(0.30, 0.80 - 0.06 * dist)
            stroke_w = max(1.2, 2.2 - 0.10 * dist)
            arc = make_arc(
                positions[i][0], positions[j][0], y_top, arc_h,
                color, opacity, stroke_w,
            )
            arcs.add(arc)

        # Self-loop
        loop = make_self_loop(positions[i], color)
        arcs.add(loop)

    return arcs


def doc_positions_centered(n=N_DOC, y=-0.6):
    """Document token positions, centered."""
    start_x = -(n - 1) * SPACING / 2
    return [np.array([start_x + i * SPACING, y, 0]) for i in range(n)]


def doc_query_positions(n_doc=N_DOC, n_query=N_QUERY, y=-0.6):
    """Document + query positions, query first then document."""
    n_total = n_query + n_doc
    start_x = -(n_total - 1) * SPACING / 2
    return [np.array([start_x + i * SPACING, y, 0]) for i in range(n_total)]


# ============================================================
# Diagram 1: Just the document — light blue, no arrows
# ============================================================
class Diagram1_Document(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        positions = doc_positions_centered()
        labels = [f"D{i}" for i in range(N_DOC)]

        squares, texts = make_tokens(positions, labels, DOC_LIGHT, DOC_BORDER_MID)
        brace, blabel = make_brace(positions, "Document", DOC_BORDER_MID)

        title = Text("Document", font_size=38, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.0)

        self.add(squares, texts, brace, blabel, title)


# ============================================================
# Diagram 2: Document with causal attention — darker blue
# ============================================================
class Diagram2_DocumentAttention(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        positions = doc_positions_centered()
        labels = [f"D{i}" for i in range(N_DOC)]

        squares, texts = make_tokens(positions, labels, DOC_MID, DOC_BORDER_MID)
        brace, blabel = make_brace(positions, "Document", DOC_BORDER_MID)

        arcs = make_causal_arcs(positions, N_DOC, ARROW_BLUE, start_idx=0)

        title = Text("Document with Causal Self-Attention", font_size=34, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.45)
        subtitle = Text(
            "Each token's representation is enriched by attending to previous tokens",
            font_size=18, color=MID_TEXT,
        )
        subtitle.move_to(DOWN * 2.8)

        self.add(squares, texts, arcs, brace, blabel, title, subtitle)


# ============================================================
# Diagram 3: Document + Query — light blue + light yellow, no arrows
# ============================================================
class Diagram3_QueryDocument(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        positions = doc_query_positions()
        q_pos = positions[:N_QUERY]
        d_pos = positions[N_QUERY:]

        q_labels = [f"Q{i}" for i in range(N_QUERY)]
        d_labels = [f"D{i}" for i in range(N_DOC)]

        q_squares, q_texts = make_tokens(q_pos, q_labels, QUERY_LIGHT, QUERY_BORDER_MID)
        d_squares, d_texts = make_tokens(d_pos, d_labels, DOC_LIGHT, DOC_BORDER_MID)

        q_brace, q_blabel = make_brace(q_pos, "Query", QUERY_BORDER_MID)
        d_brace, d_blabel = make_brace(d_pos, "Document", DOC_BORDER_MID)

        title = Text("Query + Document", font_size=38, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.0)

        self.add(q_squares, q_texts, d_squares, d_texts,
                 q_brace, q_blabel, d_brace, d_blabel, title)


# ============================================================
# Diagram 4: Query + Document with causal attention — enriched colors
# ============================================================
class Diagram4_QueryDocumentAttention(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        positions = doc_query_positions()
        q_pos = positions[:N_QUERY]
        d_pos = positions[N_QUERY:]

        q_labels = [f"Q{i}" for i in range(N_QUERY)]
        d_labels = [f"D{i}" for i in range(N_DOC)]

        # Query tokens: darker yellow (enriched from self-attention), same dark border
        q_squares, q_texts = make_tokens(q_pos, q_labels, QUERY_MID, QUERY_BORDER_MID)

        # Document tokens: blend of enriched blue (self-attention) + yellow (query influence)
        # ~55% blue, ~45% yellow — blue-dominant but clearly warmed by query
        doc_query_blend = interpolate_color(
            ManimColor(DOC_MID), ManimColor(QUERY_MID), 0.45
        )
        d_squares, d_texts = make_tokens(d_pos, d_labels, doc_query_blend, DOC_BORDER_MID)

        q_brace, q_blabel = make_brace(q_pos, "Query", QUERY_BORDER_MID)
        d_brace, d_blabel = make_brace(d_pos, "Document", DOC_BORDER_MID)

        # All causal attention arcs
        all_arcs = VGroup()
        ARC_BASE = 0.5
        ARC_STEP = 0.28
        n_total = N_QUERY + N_DOC

        for i in range(n_total):
            is_query = i < N_QUERY
            src_color = ARROW_YELLOW if is_query else ARROW_BLUE

            # Self-loop
            loop = make_self_loop(positions[i], src_color)
            all_arcs.add(loop)

            # Arcs to all previous tokens
            for j in range(i):
                dist = i - j
                arc_h = ARC_BASE + ARC_STEP * (dist - 1)
                y_top = positions[i][1] + SQUARE_SIZE / 2 + 0.06
                opacity = max(0.30, 0.80 - 0.06 * dist)
                stroke_w = max(1.2, 2.2 - 0.10 * dist)
                arc = make_arc(
                    positions[i][0], positions[j][0], y_top, arc_h,
                    src_color, opacity, stroke_w,
                )
                all_arcs.add(arc)

        title = Text("Query + Document with Causal Attention", font_size=32, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.45)
        subtitle = Text(
            "Document representations are influenced by the query through attention",
            font_size=18, color=MID_TEXT,
        )
        subtitle.move_to(DOWN * 2.8)

        self.add(q_squares, q_texts, d_squares, d_texts,
                 all_arcs,
                 q_brace, q_blabel, d_brace, d_blabel,
                 title, subtitle)


# ============================================================
# Diagram 5: Surrogate query + Document — no arrows
# ============================================================
class Diagram5_SurrogateDocument(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        n_total = N_SURROGATE + N_DOC
        start_x = -(n_total - 1) * SPACING / 2
        y = -0.2
        positions = [np.array([start_x + i * SPACING, y, 0]) for i in range(n_total)]

        s_pos = positions[:N_SURROGATE]
        d_pos = positions[N_SURROGATE:]

        s_labels = [f"S{i}" for i in range(N_SURROGATE)]
        d_labels = [f"D{i}" for i in range(N_DOC)]

        s_squares, s_texts = make_tokens(s_pos, s_labels, SURROGATE_LIGHT, SURROGATE_BORDER)
        d_squares, d_texts = make_tokens(d_pos, d_labels, DOC_LIGHT, DOC_BORDER_MID)

        s_brace, s_blabel = make_brace(s_pos, "Surrogate Query", SURROGATE_BORDER)
        d_brace, d_blabel = make_brace(d_pos, "Document", DOC_BORDER_MID)

        # Original query tokens below the surrogate row (below brace + label)
        q_y = y - SQUARE_SIZE - 1.2
        q_pos = [np.array([s_pos[i][0], q_y, 0]) for i in range(N_QUERY)]
        q_labels = [f"Q{i}" for i in range(N_QUERY)]
        q_squares, q_texts = make_tokens(q_pos, q_labels, QUERY_LIGHT, QUERY_BORDER_MID)
        q_brace, q_blabel = make_brace(q_pos, "Query", QUERY_BORDER_MID)

        title = Text("Surrogate Query + Document", font_size=38, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.0)

        self.add(s_squares, s_texts, d_squares, d_texts,
                 s_brace, s_blabel, d_brace, d_blabel,
                 q_squares, q_texts, q_brace, q_blabel,
                 title)


ARROW_ORANGE = SURROGATE_BORDER  # match surrogate border for arrows


# ============================================================
# Diagram 6: Surrogate query + Document with causal attention
# ============================================================
class Diagram6_SurrogateDocumentAttention(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        n_total = N_SURROGATE + N_DOC
        start_x = -(n_total - 1) * SPACING / 2
        y = -0.2
        positions = [np.array([start_x + i * SPACING, y, 0]) for i in range(n_total)]

        s_pos = positions[:N_SURROGATE]
        d_pos = positions[N_SURROGATE:]

        s_labels = [f"S{i}" for i in range(N_SURROGATE)]
        d_labels = [f"D{i}" for i in range(N_DOC)]

        # Surrogate: darker fill (enriched from self-attention), same orange border
        SURROGATE_MID = "#FDBA74"  # enriched coral/red
        s_squares, s_texts = make_tokens(s_pos, s_labels, SURROGATE_MID, SURROGATE_BORDER)

        # Document tokens: blend of enriched blue (self-attention) + orange (surrogate influence)
        # ~55% blue, ~45% orange — blue-dominant but clearly warmed by surrogate
        doc_surrogate_blend = interpolate_color(
            ManimColor(DOC_MID), ManimColor(SURROGATE_MID), 0.45
        )
        d_squares, d_texts = make_tokens(d_pos, d_labels, doc_surrogate_blend, DOC_BORDER_MID)

        s_brace, s_blabel = make_brace(s_pos, "Surrogate Query", SURROGATE_BORDER)
        d_brace, d_blabel = make_brace(d_pos, "Document", DOC_BORDER_MID)

        # Original query tokens below
        q_y = y - SQUARE_SIZE - 1.2
        q_pos = [np.array([s_pos[i][0], q_y, 0]) for i in range(N_QUERY)]
        q_labels = [f"Q{i}" for i in range(N_QUERY)]
        q_squares, q_texts = make_tokens(q_pos, q_labels, QUERY_LIGHT, QUERY_BORDER_MID)
        q_brace, q_blabel = make_brace(q_pos, "Query", QUERY_BORDER_MID)

        # Causal attention arcs on the main row
        all_arcs = VGroup()
        ARC_BASE = 0.5
        ARC_STEP = 0.28

        for i in range(n_total):
            is_surrogate = i < N_SURROGATE
            src_color = ARROW_ORANGE if is_surrogate else ARROW_BLUE

            # Self-loop
            loop = make_self_loop(positions[i], src_color)
            all_arcs.add(loop)

            # Arcs to all previous tokens
            for j in range(i):
                dist = i - j
                arc_h = ARC_BASE + ARC_STEP * (dist - 1)
                y_top = positions[i][1] + SQUARE_SIZE / 2 + 0.06
                opacity = max(0.30, 0.80 - 0.06 * dist)
                stroke_w = max(1.2, 2.2 - 0.10 * dist)
                arc = make_arc(
                    positions[i][0], positions[j][0], y_top, arc_h,
                    src_color, opacity, stroke_w,
                )
                all_arcs.add(arc)

        title = Text("Surrogate Query + Document with Causal Attention",
                      font_size=30, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.45)
        subtitle = Text(
            "Document representations are influenced by the surrogate query through attention",
            font_size=18, color=MID_TEXT,
        )
        subtitle.move_to(DOWN * 3.65)

        self.add(s_squares, s_texts, d_squares, d_texts,
                 all_arcs,
                 s_brace, s_blabel, d_brace, d_blabel,
                 q_squares, q_texts, q_brace, q_blabel,
                 title, subtitle)


# ============================================================
# Diagram 7: Three-row comparison — bare, query-conditioned, surrogate-conditioned
# ============================================================
def _make_tokens_sized(positions, labels, fill, border, sq_size, opacity=0.85):
    """Like make_tokens but with custom square size."""
    squares = VGroup()
    texts = VGroup()
    for pos, label in zip(positions, labels):
        sq = RoundedRectangle(
            corner_radius=0.08,
            width=sq_size, height=sq_size,
            fill_color=fill, fill_opacity=opacity,
            stroke_color=border, stroke_width=2.5,
        ).move_to(pos)
        squares.add(sq)
        txt = Text(label, font_size=18, weight=BOLD, color=DARK_TEXT)
        txt.move_to(pos)
        texts.add(txt)
    return squares, texts


class Diagram7_Comparison(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Use SAME sizing as diagrams 4 and 6 for identical token appearance
        n_full = N_QUERY + N_DOC  # 9 tokens wide
        row_start_x = -(n_full - 1) * SPACING / 2 + 1.8  # shift right for labels

        ROW_SPACING = 1.4
        row1_y = 1.5
        row2_y = row1_y - ROW_SPACING
        row3_y = row2_y - ROW_SPACING

        label_x = row_start_x - SQUARE_SIZE / 2 - 0.25

        all_objects = VGroup()

        # === Row 1: Bare document (same as Diagram 2) ===
        r1_pos = [np.array([row_start_x + (N_QUERY + i) * SPACING, row1_y, 0])
                  for i in range(N_DOC)]
        r1_sq, r1_txt = make_tokens(
            r1_pos, [f"D{i}" for i in range(N_DOC)], DOC_MID, DOC_BORDER_MID)
        all_objects.add(r1_sq, r1_txt)

        r1_label = Text("Bare document", font_size=19, color=MID_TEXT, weight=BOLD)
        r1_label.move_to([label_x, row1_y, 0]).align_to([label_x, 0, 0], RIGHT)
        all_objects.add(r1_label)

        # === Row 2: Query-conditioned (same colors as Diagram 4) ===
        r2_q_pos = [np.array([row_start_x + i * SPACING, row2_y, 0])
                    for i in range(N_QUERY)]
        r2_d_pos = [np.array([row_start_x + (N_QUERY + i) * SPACING, row2_y, 0])
                    for i in range(N_DOC)]

        r2_q_sq, r2_q_txt = make_tokens(
            r2_q_pos, [f"Q{i}" for i in range(N_QUERY)], QUERY_MID, QUERY_BORDER_MID)
        doc_query_blend = interpolate_color(
            ManimColor(DOC_MID), ManimColor(QUERY_MID), 0.45)
        r2_d_sq, r2_d_txt = make_tokens(
            r2_d_pos, [f"D{i}" for i in range(N_DOC)], doc_query_blend, DOC_BORDER_MID)
        all_objects.add(r2_q_sq, r2_q_txt, r2_d_sq, r2_d_txt)

        r2_label = Text("Query-conditioned", font_size=19, color=MID_TEXT, weight=BOLD)
        r2_label.move_to([label_x, row2_y, 0]).align_to([label_x, 0, 0], RIGHT)
        all_objects.add(r2_label)

        # === Row 3: Surrogate-conditioned (same colors as Diagram 6) ===
        r3_s_pos = [np.array([row_start_x + i * SPACING, row3_y, 0])
                    for i in range(N_SURROGATE)]
        r3_d_pos = [np.array([row_start_x + (N_QUERY + i) * SPACING, row3_y, 0])
                    for i in range(N_DOC)]

        SURROGATE_MID = "#FDBA74"
        r3_s_sq, r3_s_txt = make_tokens(
            r3_s_pos, [f"S{i}" for i in range(N_SURROGATE)], SURROGATE_MID, SURROGATE_BORDER)
        doc_surrogate_blend = interpolate_color(
            ManimColor(DOC_MID), ManimColor(SURROGATE_MID), 0.45)
        r3_d_sq, r3_d_txt = make_tokens(
            r3_d_pos, [f"D{i}" for i in range(N_DOC)], doc_surrogate_blend, DOC_BORDER_MID)
        all_objects.add(r3_s_sq, r3_s_txt, r3_d_sq, r3_d_txt)

        r3_label = Text("Surrogate-conditioned", font_size=19, color=MID_TEXT, weight=BOLD)
        r3_label.move_to([label_x, row3_y, 0]).align_to([label_x, 0, 0], RIGHT)
        all_objects.add(r3_label)

        # === Column braces at bottom ===
        brace_y = row3_y - SQUARE_SIZE / 2 - 0.1
        prefix_brace = BraceBetweenPoints(
            [row_start_x - SQUARE_SIZE / 2, brace_y, 0],
            [row_start_x + (N_QUERY - 1) * SPACING + SQUARE_SIZE / 2, brace_y, 0],
            direction=DOWN, color=MID_TEXT,
        )
        prefix_blabel = Text("Prefix", font_size=18, color=MID_TEXT, weight=BOLD)
        prefix_blabel.next_to(prefix_brace, DOWN, buff=0.08)
        all_objects.add(prefix_brace, prefix_blabel)

        doc_brace = BraceBetweenPoints(
            [row_start_x + N_QUERY * SPACING - SQUARE_SIZE / 2, brace_y, 0],
            [row_start_x + (n_full - 1) * SPACING + SQUARE_SIZE / 2, brace_y, 0],
            direction=DOWN, color=DOC_BORDER_MID,
        )
        doc_blabel = Text("Document", font_size=18, color=DOC_BORDER_MID, weight=BOLD)
        doc_blabel.next_to(doc_brace, DOWN, buff=0.08)
        all_objects.add(doc_brace, doc_blabel)

        # === Title ===
        title = Text("Document Representations Under Different Conditioning",
                      font_size=28, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.3)
        all_objects.add(title)

        self.add(all_objects)


# ============================================================
# Diagram 8: Many queries → few documents, color-grouped
# ============================================================
class Diagram8_QueryDiversity(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # 4 groups, each with 9 queries (3x3 grid), a surrogate, and a document
        # Color endpoints per group: light → dark within a hue family
        group_defs = [
            {"light": "#FECACA", "dark": "#B91C1C", "border": "#DC2626", "mid": "#F87171"},
            {"light": "#FED7AA", "dark": "#C2410C", "border": "#EA580C", "mid": "#FB923C"},
            {"light": "#FEF3C7", "dark": "#92400E", "border": "#D97706", "mid": "#FBBF24"},
            {"light": "#D1FAE5", "dark": "#065F46", "border": "#16A34A", "mid": "#4ADE80"},
        ]

        N_PER_GROUP = 9  # 3x3 grid
        GRID_COLS = 3
        GRID_ROWS = 3
        SQ = 0.34           # small query squares
        CELL = 0.40         # grid cell spacing
        SQ_SUR = 0.52       # surrogate square (larger)
        SQ_DOC = 0.55       # document square
        GRID_X = -3.3       # left edge of query grid
        SUR_X = 0.3         # surrogate column
        DOC_X = 3.0         # document column
        GROUP_GAP = 1.75    # vertical distance between group centers
        top_y = 2.4         # y of first group center

        doc_base = ManimColor(DOC_MID)
        all_objects = VGroup()

        for g_idx, gdef in enumerate(group_defs):
            gy = top_y - g_idx * GROUP_GAP
            light = ManimColor(gdef["light"])
            dark = ManimColor(gdef["dark"])
            border_c = gdef["border"]
            mid_c = gdef["mid"]

            # Generate 9 fill colors as a gradient light → dark
            fills = [interpolate_color(light, dark, i / (N_PER_GROUP - 1))
                     for i in range(N_PER_GROUP)]

            # --- 3x3 query grid ---
            grid_center_x = GRID_X + (GRID_COLS - 1) * CELL / 2
            q_positions = []
            q_num_base = g_idx * N_PER_GROUP + 1
            for idx in range(N_PER_GROUP):
                row = idx // GRID_COLS
                col = idx % GRID_COLS
                qx = GRID_X + col * CELL
                qy = gy + (1 - row) * CELL  # top row first
                q_pos = np.array([qx, qy, 0])
                q_positions.append(q_pos)

                sq = RoundedRectangle(
                    corner_radius=0.06, width=SQ, height=SQ,
                    fill_color=fills[idx], fill_opacity=0.85,
                    stroke_color=border_c, stroke_width=1.8,
                ).move_to(q_pos)
                all_objects.add(sq)

            # --- Surrogate (average) query ---
            sur_pos = np.array([SUR_X, gy, 0])
            # Surrogate fill = midpoint of the gradient (the "average")
            sur_fill = interpolate_color(light, dark, 0.5)
            sur_sq = RoundedRectangle(
                corner_radius=0.08, width=SQ_SUR, height=SQ_SUR,
                fill_color=sur_fill, fill_opacity=0.90,
                stroke_color=border_c, stroke_width=2.5,
            ).move_to(sur_pos)
            sur_txt = Text(f"S{g_idx+1}", font_size=14, weight=BOLD, color=DARK_TEXT)
            sur_txt.move_to(sur_pos)
            all_objects.add(sur_sq, sur_txt)

            # --- Arrows: queries → surrogate (thin, faint) ---
            for q_pos in q_positions:
                start = np.array([q_pos[0] + SQ / 2 + 0.04, q_pos[1], 0])
                end = np.array([SUR_X - SQ_SUR / 2 - 0.04, gy, 0])
                line = Line(start, end, color=border_c, stroke_width=0.8, stroke_opacity=0.20)
                all_objects.add(line)

            # --- Document ---
            d_pos = np.array([DOC_X, gy, 0])
            doc_blend = interpolate_color(doc_base, ManimColor(mid_c), 0.40)
            d_sq = RoundedRectangle(
                corner_radius=0.08, width=SQ_DOC, height=SQ_DOC,
                fill_color=doc_blend, fill_opacity=0.85,
                stroke_color=DOC_BORDER_MID, stroke_width=2.5,
            ).move_to(d_pos)
            d_txt = Text(f"D{g_idx+1}", font_size=16, weight=BOLD, color=DARK_TEXT)
            d_txt.move_to(d_pos)
            all_objects.add(d_sq, d_txt)

            # --- Arrow: surrogate → document (thicker, more visible) ---
            s_start = np.array([SUR_X + SQ_SUR / 2 + 0.06, gy, 0])
            s_end = np.array([DOC_X - SQ_DOC / 2 - 0.06, gy, 0])
            sur_line = Line(s_start, s_end, color=border_c, stroke_width=2.2, stroke_opacity=0.5)
            # Arrowhead
            direction = s_end - s_start
            direction = direction / np.linalg.norm(direction)
            perp = np.array([-direction[1], direction[0], 0])
            tip_sz = 0.09
            sur_tip = Polygon(
                s_end,
                s_end - direction * tip_sz * 1.5 + perp * tip_sz * 0.6,
                s_end - direction * tip_sz * 1.5 - perp * tip_sz * 0.6,
                fill_color=border_c, fill_opacity=0.5, stroke_width=0,
            )
            all_objects.add(sur_line, sur_tip)

        # --- Column labels ---
        header_y = top_y + CELL + 0.55
        q_label = Text("Queries", font_size=20, color=DARK_TEXT, weight=BOLD)
        q_label.move_to([GRID_X + (GRID_COLS - 1) * CELL / 2, header_y, 0])
        all_objects.add(q_label)

        s_label = Text("Surrogate", font_size=20, color=MID_TEXT, weight=BOLD)
        s_label.move_to([SUR_X, header_y, 0])
        all_objects.add(s_label)

        d_label = Text("Document", font_size=20, color=DOC_BORDER_MID, weight=BOLD)
        d_label.move_to([DOC_X, header_y, 0])
        all_objects.add(d_label)

        # --- Title ---
        title = Text("Many Queries Compress into Surrogates",
                      font_size=26, color=DARK_TEXT, weight=BOLD)
        title.move_to([0, header_y + 0.55, 0])
        all_objects.add(title)

        self.add(all_objects)
