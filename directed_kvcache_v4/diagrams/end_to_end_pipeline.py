from manim import *
import numpy as np

# ── Canonical palette (matches attention_progression.py) ──────────
DARK = "#1a1a1a"
MID = "#555555"
FAINT = "#aaaaaa"

DOC_LIGHT, DOC_MID    = "#BFDBFE", "#60A5FA"
DOC_BD_L,  DOC_BD_M   = "#93C5FD", "#3B82F6"
Q_LIGHT,   Q_BD       = "#FDE68A", "#D97706"
S_LIGHT,   S_BD       = "#FED7AA", "#EA580C"
BOS_F,     BOS_B      = "#E5E7EB", "#9CA3AF"
NL_F,      NL_B       = "#F3F4F6", "#9CA3AF"
A_F, A_B, A_ACC       = "#FECACA", "#EF4444", "#DC2626"

GREEN, PURPLE, BLUE, RED = "#059669", "#7C3AED", "#3B82F6", "#EF4444"

# Font used everywhere — DejaVu Sans has clean kerning at all sizes
FONT = "DejaVu Sans"


def T(text, **kw):
    """Text with consistent font."""
    kw.setdefault("font", FONT)
    kw.setdefault("color", DARK)
    return Text(text, **kw)


def tok(label, fill, border, fs=14, op=0.85):
    t = T(label, font_size=fs)
    w = max(t.width + 0.30, 0.52)
    r = RoundedRectangle(corner_radius=0.07, width=w, height=0.44,
                         fill_color=fill, fill_opacity=op,
                         stroke_color=border, stroke_width=2.4)
    return VGroup(r, t), w


def toks(x, y, labels, fill, border, gap=0.10, fs=14):
    g = VGroup()
    for lab in labels:
        t, w = tok(lab, fill, border, fs=fs)
        t.move_to([x + w / 2, y, 0])
        g.add(t)
        x += w + gap
    return g, x


def rbox(cx, cy, w, h, fill, border, op=0.14):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h,
                            fill_color=fill, fill_opacity=op,
                            stroke_color=border, stroke_width=2.0
                            ).move_to([cx, cy, 0])


def darr(y1, y2, color=BLUE, sw=2.2):
    return Arrow([0, y1, 0], [0, y2, 0], color=color,
                 stroke_width=sw, buff=0,
                 max_tip_length_to_length_ratio=0.25)


def left_text(text, x, y, **kw):
    """Text with its LEFT edge at x."""
    t = T(text, **kw)
    t.move_to([x + t.width / 2, y, 0])
    return t


class EndToEndPipeline(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        self.camera.frame_height = 10
        self.camera.frame_width = 14.22 * 10 / 8

        # ── Layout constants ──────────────────────────────────
        LEFT = -5.6          # left edge for all rows + headers

        Y_TITLE   =  4.30
        Y_SUB     =  3.82
        Y_PA_HDR  =  3.18
        Y_PA_TOKS =  2.55
        Y_PA_LBL  =  2.08
        Y_CACHE_H =  1.48
        Y_CACHE   =  1.08
        Y_SEL     =  0.58
        Y_ENR_H   =  0.25
        Y_ENR     = -0.10
        Y_ENR_POS = -0.52
        Y_ENR_NOTE= -0.88
        Y_DIV     = -1.28
        Y_PB_HDR  = -1.65
        Y_PB_TOKS = -2.25
        Y_PB_LBL  = -2.68
        Y_NLL     = -3.60

        G  = 0.10            # intra-group token gap
        SG = 0.22            # inter-group token gap

        # ── Font size tiers ───────────────────────────────────
        FS_TITLE   = 34
        FS_SUB     = 15
        FS_PHASE   = 20      # "Phase A / B" headers
        FS_SECTION = 15      # "Full KV Cache", "Enriched Cache"
        FS_TOK     = 14      # Phase A tokens
        FS_TOK_S   = 13      # cache rows + Phase B tokens
        FS_LABEL   = 13      # group labels under tokens
        FS_ANNOT   = 12      # annotations, italic notes
        FS_POS     = 10      # position labels
        FS_NLL_T   = 18      # NLL box title
        FS_NLL_D   = 16      # NLL box data lines
        FS_NLL_G   = 17      # NLL box delta

        # ═════════════════════════════════════════════════════════
        # TITLE
        # ═════════════════════════════════════════════════════════
        self.add(T("End-to-End Pipeline: Extract Condition",
                   font_size=FS_TITLE, weight=BOLD
                   ).move_to([0, Y_TITLE, 0]))
        self.add(T('Prefix = "Extract all key data points, facts, entities, '
                   'and specific attributes..."',
                   font_size=FS_SUB, color=MID, slant=ITALIC
                   ).move_to([0, Y_SUB, 0]))

        # ═════════════════════════════════════════════════════════
        # PHASE A
        # ═════════════════════════════════════════════════════════
        self.add(left_text("Phase A  \u2014  Conditioning", LEFT, Y_PA_HDR,
                           font_size=FS_PHASE, weight=BOLD, color=GREEN))

        x = LEFT
        g_bos, x = toks(x, Y_PA_TOKS, ["BOS"], BOS_F, BOS_B, gap=SG, fs=FS_TOK)
        self.add(g_bos)

        g_pfx, x = toks(x, Y_PA_TOKS,
                         ["Extract", "all", "key", "data", "...", "text"],
                         S_LIGHT, S_BD, gap=G, fs=FS_TOK)
        self.add(g_pfx)
        x += SG - G

        g_nl, x = toks(x, Y_PA_TOKS, ["\\n"], NL_F, NL_B, gap=SG, fs=FS_TOK)
        self.add(g_nl)

        g_doc, x = toks(x, Y_PA_TOKS,
                         ["The", "Eiffel", "Tower", "...", "1889", "..."],
                         DOC_LIGHT, DOC_BD_L, gap=G, fs=FS_TOK)
        self.add(g_doc)

        for grp, label, color in [
            (g_pfx, "prefix (P tokens)", S_BD),
            (g_doc, "document (D tokens)", DOC_BD_M),
        ]:
            cx = (grp[0].get_left()[0] + grp[-1].get_right()[0]) / 2
            self.add(T(label, font_size=FS_LABEL, color=color, weight=BOLD
                       ).move_to([cx, Y_PA_LBL, 0]))

        # ═════════════════════════════════════════════════════════
        # FULL KV CACHE
        # ═════════════════════════════════════════════════════════
        self.add(darr(Y_PA_LBL - 0.22, Y_CACHE_H + 0.18, GREEN))

        self.add(left_text("Full KV Cache", LEFT, Y_CACHE_H,
                           font_size=FS_SECTION, weight=BOLD, color=BLUE))

        cx = LEFT
        cb, cx = toks(cx, Y_CACHE, ["BOS"], BOS_F, BOS_B, gap=SG, fs=FS_TOK_S)
        self.add(cb)

        cp, cx = toks(cx, Y_CACHE, ["Ext", "all", "key", "...", "text"],
                       S_LIGHT, S_BD, gap=0.06, fs=FS_TOK_S)
        for t in cp:
            t[0].set_fill(opacity=0.25)
            t[1].set_opacity(0.28)
        self.add(cp)
        cx += 0.06

        cn, cx = toks(cx, Y_CACHE, ["\\n"], NL_F, NL_B, gap=SG, fs=FS_TOK_S)
        cn[0][0].set_fill(opacity=0.25)
        cn[0][1].set_opacity(0.28)
        self.add(cn)

        cd, cx = toks(cx, Y_CACHE,
                       ["The", "Eiffel", "Tower", "...", "1889", "..."],
                       DOC_MID, DOC_BD_M, gap=0.06, fs=FS_TOK_S)
        self.add(cd)

        sl = cp[0].get_left()[0] - 0.08
        sr = cn[0].get_right()[0] + 0.08
        self.add(Line([sl, Y_CACHE, 0], [sr, Y_CACHE, 0],
                      color=RED, stroke_width=3.5, stroke_opacity=0.8))

        self.add(T("select_kv_cache() \u2014 remove prefix",
                   font_size=FS_ANNOT, color=RED, weight=BOLD
                   ).move_to([(sl + sr) / 2, Y_SEL, 0]))

        # ═════════════════════════════════════════════════════════
        # ENRICHED CACHE
        # ═════════════════════════════════════════════════════════
        self.add(darr(Y_SEL - 0.20, Y_ENR_H + 0.18, BLUE))

        repo_mid_y = (Y_SEL - 0.20 + Y_ENR_H + 0.18) / 2
        self.add(T("reposition_kv_cache()",
                   font_size=11, color=BLUE, slant=ITALIC
                   ).move_to([3.0, repo_mid_y + 0.10, 0]))
        self.add(T("RoPE-rotate \u2192 pos 1..D",
                   font_size=FS_POS, color=BLUE, slant=ITALIC
                   ).move_to([3.0, repo_mid_y - 0.14, 0]))

        self.add(left_text("Enriched Cache", LEFT, Y_ENR_H,
                           font_size=FS_SECTION, weight=BOLD, color=DOC_BD_M))

        ex = LEFT
        eb, ex = toks(ex, Y_ENR, ["BOS"], BOS_F, BOS_B, gap=SG, fs=FS_TOK_S)
        self.add(eb)

        ed, ex = toks(ex, Y_ENR,
                       ["The'", "Eiffel'", "Tower'", "...'", "1889'", "...'"],
                       DOC_MID, DOC_BD_M, gap=0.06, fs=FS_TOK_S)
        self.add(ed)

        for t, p in zip(
            [eb[0]] + list(ed),
            ["pos 0", "pos 1", "pos 2", "pos 3", "...", "pos D\u22121", "pos D"],
        ):
            self.add(T(p, font_size=FS_POS, color=FAINT
                       ).move_to([t.get_center()[0], Y_ENR_POS, 0]))

        self.add(T("Doc representations enriched by prefix attention "
                   "(prefix itself discarded)",
                   font_size=FS_ANNOT, color=DOC_BD_M, slant=ITALIC
                   ).move_to([0, Y_ENR_NOTE, 0]))

        # ═════════════════════════════════════════════════════════
        # DIVIDER
        # ═════════════════════════════════════════════════════════
        self.add(DashedLine([-7, Y_DIV, 0], [7, Y_DIV, 0],
                            color="#D1D5DB", stroke_width=1.5,
                            dash_length=0.15))

        # ═════════════════════════════════════════════════════════
        # PHASE B
        # ═════════════════════════════════════════════════════════
        self.add(left_text("Phase B  \u2014  Inference (what the model sees)",
                           LEFT, Y_PB_HDR,
                           font_size=FS_PHASE, weight=BOLD, color=PURPLE))

        bx = LEFT
        b_bos, bx = toks(bx, Y_PB_TOKS, ["BOS"],
                          BOS_F, BOS_B, gap=G, fs=FS_TOK_S)
        self.add(b_bos)

        b_cache, bx = toks(bx, Y_PB_TOKS,
                            ["The'", "Eiffel'", "...'", "1889'"],
                            DOC_MID, DOC_BD_M, gap=G, fs=FS_TOK_S)
        self.add(b_cache)
        bx += SG - G

        bn1, bx = toks(bx, Y_PB_TOKS, ["\\n"], NL_F, NL_B, gap=SG, fs=FS_TOK_S)
        self.add(bn1)

        bq, bx = toks(bx, Y_PB_TOKS,
                        ["When", "was", "the", "Eiffel", "Tower",
                         "completed", "?"],
                        Q_LIGHT, Q_BD, gap=G, fs=FS_TOK_S)
        self.add(bq)
        bx += SG - G

        bn2, bx = toks(bx, Y_PB_TOKS, ["\\n"], NL_F, NL_B, gap=SG, fs=FS_TOK_S)
        self.add(bn2)

        ba, bx = toks(bx, Y_PB_TOKS, ["1889"], A_F, A_B, gap=G, fs=FS_TOK_S)
        self.add(ba)

        cache_cx = (b_bos[0].get_left()[0] + b_cache[-1].get_right()[0]) / 2
        self.add(T("enriched cache", font_size=FS_LABEL, color=DOC_BD_M,
                   weight=BOLD).move_to([cache_cx, Y_PB_LBL, 0]))
        qcx = (bq[0].get_left()[0] + bq[-1].get_right()[0]) / 2
        self.add(T("query", font_size=FS_LABEL, color=Q_BD, weight=BOLD
                   ).move_to([qcx, Y_PB_LBL, 0]))
        acx = ba[0].get_center()[0]
        self.add(T("answer", font_size=FS_LABEL, color=A_ACC, weight=BOLD
                   ).move_to([acx, Y_PB_LBL, 0]))

        # ═════════════════════════════════════════════════════════
        # NLL RESULT
        # ═════════════════════════════════════════════════════════
        self.add(darr(Y_PB_LBL - 0.22, Y_NLL + 0.65, PURPLE))

        nll_box = rbox(0, Y_NLL, 6.0, 1.35, "#FFFFFF", "#D1D5DB", op=0.0)
        self.add(nll_box)

        # Title
        self.add(T('Predicting "1889"',
                   font_size=FS_NLL_T, weight=BOLD
                   ).move_to([0, Y_NLL + 0.44, 0]))

        # Data lines — left-aligned
        NLL_L = -1.7
        NLL_V =  1.0
        r1_y = Y_NLL + 0.10
        r2_y = Y_NLL - 0.18

        self.add(left_text("Enriched cache:", NLL_L, r1_y,
                           font_size=FS_NLL_D, color=DARK))
        self.add(left_text("2.14 nats", NLL_V, r1_y,
                           font_size=FS_NLL_D, weight=BOLD))

        self.add(left_text("Bare cache:", NLL_L, r2_y,
                           font_size=FS_NLL_D, color=MID))
        self.add(left_text("2.83 nats", NLL_V, r2_y,
                           font_size=FS_NLL_D, color=MID))

        # Delta
        self.add(T("\u0394 = \u22120.69 nats",
                   font_size=FS_NLL_G, color=GREEN, weight=BOLD
                   ).move_to([0, Y_NLL - 0.48, 0]))

        # ═════════════════════════════════════════════════════════
        # RIGHT SIDE — past_key_values connector
        # ═════════════════════════════════════════════════════════
        rx = 6.8
        ct, cb = Y_ENR, Y_PB_TOKS
        for y in [ct, cb]:
            self.add(Line([rx - 0.18, y, 0], [rx + 0.18, y, 0],
                          color=BLUE, stroke_width=2.0, stroke_opacity=0.5))
        self.add(Line([rx, ct, 0], [rx, cb, 0],
                      color=BLUE, stroke_width=2.0, stroke_opacity=0.5))
        lbl = T("past_key_values", font_size=FS_ANNOT, color=BLUE,
                weight=BOLD, slant=ITALIC)
        lbl.move_to([rx, (ct + cb) / 2, 0]).rotate(PI / 2)
        self.add(lbl)
