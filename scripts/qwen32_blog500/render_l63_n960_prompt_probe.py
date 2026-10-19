#!/usr/bin/env python3
"""Render the already-measured L63/N960 prompt-probe results."""

from pathlib import Path


OUTPUT = Path("outputs/blog500_pile1m/l63_n960_prompt_probe.html")

ROWS = [
    ("Slashdot extended maximizer", 62, 102.0, " -", -8.9375, " the", "Stained glass assembly transforms light into color—geometry becomes spirituality. #NetNeutrality is STILL in danger - Click here to help. DEAL: For $28 - Lifetime Glass Design Library! Use promo code SLASHDOT28. Check out the new SourceForge HTML5 Internet speed test. ×", [(" DEAL", .492188),(":",7.625),(" For",15.5625),(" $",-1.390625),("2",1.59375),("8",.099121),(" -",102.0),(" Lifetime",.550781),(" Glass",.417969),(" Design",-1.75),(" Library",4.78125)]),
    ("Slashdot deal footer", 27, 96.0, " -", -1.335938, " Library", "#NetNeutrality is STILL in danger - Click here to help. DEAL: For $28 - Lifetime Glass Design Library!", [(" DEAL",.351562),(":",3.53125),(" For",5.875),(" $",-.441406),("2",1.882812),("8",-.953125),(" -",96.0),(" Lifetime",.236328),(" Glass",.421875),(" Design",.175781),(" Library",-1.335938)]),
    ("Gaokao / Elective 4", 61, 81.0, "4", -1.429688, " In", "Elective 4-5: Special Topics on Inequalities\n\nThe inequality $|x+2|+|x-1| > a^2 - 2a$ holds for all $x \\in \\mathbb{R}$. Find the range of the real number $a$.", [("Elect",-.141602),("ive",-.155273),(" ",-1.226562),("4",81.0),("-",3.34375),("5",6.78125),(":",-.244141),(" Special",-.609375),(" Topics",3.9375),(" on",-.53125),(" In",-1.429688),("equal",.289062),("ities",13.0625)]),
    ("Gaokao blind-max header", 14, 81.0, "4", -1.226562, " ", "Elective 4 Unit 1 Great Scientists Period One Warming Up", [("Elect",-.141602),("ive",-.155273),(" ",-1.226562),("4",81.0),(" Unit",.257812),(" ",-.078613),("1",-.014221),(" Great",-.455078),(" Scientists",-.114746),(" Period",-.033691),(" One",-.460938),(" W",-.003906),("arming",-.224609),(" Up",-.126953)]),
    ("Baroque encyclopedia lede", 46, 63.25, " some", -3.6875, " and", "The Baroque is a style of architecture originating in late 16th-century Italy and lasting in some regions, notably Germany and colonial South America, until the 18th century, characterised by dramatic contrast and ornament.", [(" Italy",-.144531),(" and",1.570312),(" lasting",-.135742),(" in",6.84375),(" some",63.25),(" regions",-.488281),(",",8.8125),(" notably",29.25),(" Germany",1.453125),(" and",-3.6875),(" colonial",-.066895),(" South",.925781),(" America",6.625),(",",8.25)]),
    ("OUDCE course boilerplate", 12, 1.085938, " to", -.582031, " using", "To participate you must to be familiar with using a computer.", [("To",-.065918),(" participate",-.484375),(" you",-.071289),(" must",.202148),(" to",1.085938),(" be",-.289062),(" familiar",.03418),(" with",-.558594),(" using",-.582031),(" a",.535156),(" computer",-.036621),(".",-.012024)]),
]


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    cards = []
    for name, count, maximum, peak, minimum, trough, prompt, trace in ROWS:
        tokens = "".join(
            f'<span class="tok {"pos" if value >= 0 else "neg"}" '
            f'style="--a:{min(.9, abs(value) / 102):.3f}">'
            f'{esc(piece).replace(" ", "·")}<b>{value:+.3f}</b></span>'
            for piece, value in trace
        )
        cards.append(f"""<article><h2>{esc(name)}</h2><pre>{esc(prompt)}</pre>
        <div class="stats"><span>Tokens <b>{count}</b></span><span>Maximum <b>{maximum:+.3f}</b> on <code>{esc(repr(peak))}</code></span><span>Minimum <b>{minimum:+.3f}</b> on <code>{esc(repr(trough))}</code></span></div>
        <h3>Measured token trace around the response</h3><div class="trace">{tokens}</div></article>""")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L63/N960 prompt probe</title><style>
    :root{{--ink:#182027;--muted:#66727d;--line:#dce3e8;--paper:#fff;--wash:#f4f7f8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}header{{padding:24px 5vw;background:white;border-bottom:1px solid var(--line)}}h1{{margin:0 0 5px}}header p{{margin:0;color:var(--muted)}}main{{max-width:1180px;margin:auto;padding:20px}}article{{margin-bottom:16px;padding:18px;border:1px solid var(--line);border-radius:10px;background:white}}h2{{margin:0 0 10px}}h3{{margin:15px 0 7px;font-size:13px}}pre{{padding:12px;border-radius:7px;background:#f7f9fa;white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace}}.stats{{display:flex;flex-wrap:wrap;gap:8px}}.stats span{{padding:7px 9px;border:1px solid var(--line);border-radius:6px}}.trace{{display:flex;flex-wrap:wrap;gap:5px}}.tok{{padding:5px 7px;border-radius:5px;font:12px ui-monospace,monospace}}.tok b{{display:block;font-size:10px;text-align:center}}.pos{{background:rgba(213,75,35,calc(.08 + var(--a)))}}.neg{{background:rgba(37,99,165,calc(.08 + var(--a)))}}.note{{max-width:1180px;margin:18px auto 0;padding:0 20px;color:var(--muted)}}code{{font-family:ui-monospace,monospace}}</style></head><body>
    <header><h1>Layer 63 · Neuron 960 — positive prompt probe</h1><p>Qwen3-32B · post-SwiGLU / pre-down-projection activation · maxima and minima computed over every token in each complete prompt</p></header>
    <p class="note">Warm tokens are positive and cool tokens are negative. Each trace shows the informative response region; the maximum and minimum summaries were computed over the entire prompt.</p><main>{''.join(cards)}</main></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT.resolve()}")
    print(f"Prompt results: {len(ROWS)}")


if __name__ == "__main__":
    main()
