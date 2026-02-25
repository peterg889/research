from manim import *
import numpy as np


# Richer, saturated colors that pop on white
PREFIX_YELLOW = "#E8A817"
PREFIX_FILL = "#FCD34D"
DOC_BLUE = "#2563EB"
DOC_FILL = "#60A5FA"
DARK_TEXT = "#1a1a1a"
MID_TEXT = "#555555"


class CausalAttentionDiagram(Scene):
    def construct(self):
        # White background
        self.camera.background_color = WHITE

        # Configuration
        N_PREFIX = 4
        N_DOC = 4
        N_TOTAL = N_PREFIX + N_DOC
        SQUARE_SIZE = 0.65
        SPACING = 0.95

        prefix_labels = [f"Q{i}" for i in range(N_PREFIX)]
        doc_labels = [f"D{i}" for i in range(N_DOC)]
        all_labels = prefix_labels + doc_labels

        start_x = -(N_TOTAL - 1) * SPACING / 2
        token_y = -1.0
        positions = [np.array([start_x + i * SPACING, token_y, 0]) for i in range(N_TOTAL)]

        all_objects = VGroup()

        # --- Token squares ---
        squares = VGroup()
        token_texts = VGroup()
        for i in range(N_TOTAL):
            is_prefix = i < N_PREFIX
            fill = PREFIX_FILL if is_prefix else DOC_FILL
            border = PREFIX_YELLOW if is_prefix else DOC_BLUE

            sq = RoundedRectangle(
                corner_radius=0.08,
                width=SQUARE_SIZE,
                height=SQUARE_SIZE,
                fill_color=fill,
                fill_opacity=0.85,
                stroke_color=border,
                stroke_width=2.5,
            ).move_to(positions[i])
            squares.add(sq)

            txt = Text(
                all_labels[i], font_size=20, weight=BOLD,
                color=DARK_TEXT,
            )
            txt.move_to(positions[i])
            token_texts.add(txt)

        all_objects.add(squares, token_texts)

        # --- Causal attention arcs ---
        # Arrow color = color of the SOURCE token (the one attending)
        arcs = VGroup()
        ARC_BASE_HEIGHT = 0.5
        ARC_HEIGHT_PER_STEP = 0.28

        for i in range(1, N_TOTAL):
            # Color based on source token
            src_color = PREFIX_YELLOW if i < N_PREFIX else DOC_BLUE

            for j in range(i):
                dist = i - j
                arc_h = ARC_BASE_HEIGHT + ARC_HEIGHT_PER_STEP * (dist - 1)

                x_start = positions[i][0]
                x_end = positions[j][0]
                y_top = token_y + SQUARE_SIZE / 2 + 0.06

                x_mid = (x_start + x_end) / 2
                y_mid = y_top + arc_h

                # Fade longer arcs gently
                opacity = max(0.30, 0.80 - 0.06 * dist)
                stroke_w = max(1.2, 2.2 - 0.10 * dist)

                # Quadratic bezier arc
                n_points = 30
                ts = np.linspace(0, 1, n_points)
                points = []
                for t in ts:
                    px = (1 - t)**2 * x_start + 2*(1 - t)*t * x_mid + t**2 * x_end
                    py = (1 - t)**2 * y_top + 2*(1 - t)*t * y_mid + t**2 * y_top
                    points.append([px, py, 0])

                curve = VMobject(
                    color=src_color,
                    stroke_width=stroke_w,
                    stroke_opacity=opacity,
                )
                curve.set_points_smoothly([np.array(p) for p in points])

                # Arrowhead triangle at the destination
                end_pt = np.array(points[-1])
                prev_pt = np.array(points[-3])
                direction = end_pt - prev_pt
                direction = direction / np.linalg.norm(direction)
                perp = np.array([-direction[1], direction[0], 0])
                tip_size = 0.09
                tip_tri = Polygon(
                    end_pt,
                    end_pt - direction * tip_size * 1.5 + perp * tip_size * 0.6,
                    end_pt - direction * tip_size * 1.5 - perp * tip_size * 0.6,
                    fill_color=src_color,
                    fill_opacity=opacity,
                    stroke_width=0,
                )
                arcs.add(curve, tip_tri)

        all_objects.add(arcs)

        # --- Self-attention: arrow that loops back to the SAME token ---
        # Short arc that starts at top-right of the square, arcs up, and lands at top-left
        self_loops = VGroup()
        for i in range(N_TOTAL):
            color = PREFIX_YELLOW if i < N_PREFIX else DOC_BLUE
            cx = positions[i][0]
            top_y = token_y + SQUARE_SIZE / 2 + 0.06

            # Start slightly right of center, end slightly left
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

            loop_curve = VMobject(
                color=color,
                stroke_width=1.8,
                stroke_opacity=0.6,
            )
            loop_curve.set_points_smoothly([np.array(p) for p in pts])

            # Tiny arrowhead
            end_pt = np.array(pts[-1])
            prev_pt = np.array(pts[-2])
            d = end_pt - prev_pt
            d = d / np.linalg.norm(d)
            perp = np.array([-d[1], d[0], 0])
            ts2 = 0.065
            tip_tri = Polygon(
                end_pt,
                end_pt - d * ts2 * 1.5 + perp * ts2 * 0.6,
                end_pt - d * ts2 * 1.5 - perp * ts2 * 0.6,
                fill_color=color,
                fill_opacity=0.6,
                stroke_width=0,
            )
            self_loops.add(loop_curve, tip_tri)

        all_objects.add(self_loops)

        # --- Braces below tokens ---
        brace_y = token_y - SQUARE_SIZE / 2 - 0.12

        prefix_brace = BraceBetweenPoints(
            [positions[0][0] - SQUARE_SIZE / 2, brace_y, 0],
            [positions[N_PREFIX - 1][0] + SQUARE_SIZE / 2, brace_y, 0],
            direction=DOWN, color=PREFIX_YELLOW,
        )
        prefix_label = Text("Prefix", font_size=22, color=PREFIX_YELLOW, weight=BOLD)
        prefix_label.next_to(prefix_brace, DOWN, buff=0.1)

        doc_brace = BraceBetweenPoints(
            [positions[N_PREFIX][0] - SQUARE_SIZE / 2, brace_y, 0],
            [positions[N_TOTAL - 1][0] + SQUARE_SIZE / 2, brace_y, 0],
            direction=DOWN, color=DOC_BLUE,
        )
        doc_label = Text("Document", font_size=22, color=DOC_BLUE, weight=BOLD)
        doc_label.next_to(doc_brace, DOWN, buff=0.1)

        all_objects.add(prefix_brace, prefix_label, doc_brace, doc_label)

        # --- Title ---
        title = Text("Causal Attention", font_size=38, color=DARK_TEXT, weight=BOLD)
        title.move_to(UP * 3.45)
        all_objects.add(title)

        # --- Subtitle ---
        subtitle = Text(
            "Each token attends to itself and all previous tokens",
            font_size=20, color=MID_TEXT,
        )
        subtitle.move_to(DOWN * 3.2)
        all_objects.add(subtitle)

        self.add(all_objects)
