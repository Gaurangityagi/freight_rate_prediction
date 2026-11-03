"""Builds report/freight_rate_report.pdf from saved outputs. Run after src.backtest and src.run_pipeline."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
HERE = ROOT / "report"
INK, MUTED, TEAL, GREY, RULE = "#1F2A2E", "#55666B", "#064A56", "#A9B7BA", "#D9E2E4"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def quote_signal_figure():
    data = pd.read_csv(OUT / "quote_signal_by_month.csv")
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    figure, axis = plt.subplots(figsize=(7.2, 2.2), dpi=200)
    bar_colors = [TEAL if f == "train" else "#C8553D" for f in data["file"]]
    axis.bar(labels, data["corr"], color=bar_colors, width=0.62)
    axis.axhline(0, color=GREY, linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Correlation with $/mile proxy")
    axis.text(10.5, 0.35, "Validation:\nno signal", ha="center", color="#C8553D", fontsize=8.5)
    axis.grid(axis="y", color=RULE, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    path = HERE / "fig_quote_signal.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def time_figure():
    daily = pd.read_csv(OUT / "daily_time_effect.csv", parse_dates=["date"])
    figure, axis = plt.subplots(figsize=(7.2, 2.3), dpi=200)
    axis.plot(daily["date"], daily["observed"] * 100, color=GREY, linewidth=1.6, label="Observed daily level")
    axis.plot(daily["date"], daily["fitted"] * 100, color=TEAL, linewidth=1.1, label="Fitted time model")
    axis.set_ylabel("Effect on rate (%, approx.)")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.grid(axis="y", color=RULE, linewidth=0.6)
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    figure.tight_layout()
    path = HERE / "fig_time_effect.png"
    figure.savefig(path)
    plt.close(figure)
    return path


body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=13.6, textColor=colors.HexColor(INK), spaceAfter=6, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor(MUTED))
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11, spaceAfter=0)
head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11.4, leading=15, spaceBefore=9, spaceAfter=4, textColor=colors.HexColor(TEAL))


def table(rows, widths):
    content = [[Paragraph(str(v), head if i == 0 else cell) for v in row] for i, row in enumerate(rows)]
    result = Table(content, colWidths=[w * 0.97 * mm for w in widths], repeatRows=1)
    result.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(TEAL)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return result


def picture(path, width_mm=168):
    from PIL import Image as Pil
    w, h = Pil.open(path).size
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def build():
    summary = pd.read_csv(OUT / "walk_forward_summary.csv").set_index("model")
    folds = pd.read_csv(OUT / "walk_forward_folds.csv")
    final = folds[folds["model"] == "two-stage (final)"].set_index("cutoff")
    trap = pd.read_csv(OUT / "quote_signal_experiments.csv").set_index("experiment")
    cold = pd.read_csv(OUT / "cold_city_holdout.csv").set_index("segment")
    coef = json.loads((OUT / "time_coefficients.json").read_text())
    ramp = pd.read_csv(OUT / "december_ramp_sensitivity.csv")
    best = summary.loc["two-stage (final)"]
    last = ramp.iloc[-1]

    s = []
    s += [Paragraph("Freight Rate Prediction: Validation Report", h1),
          Paragraph("Harshit | Machine Learning Engineer assessment", small), Spacer(1, 6)]
    s += [Paragraph(
        f"The final model is a linear time model plus LightGBM, trained on about 47,300 clean loads. On four forward-in-time "
        f"tests it averaged <b>{best.mape_clean:.2f}% MAPE (${best.mae_clean:.0f} MAE)</b> on rows with sound labels, and "
        f"{best.mape_all:.2f}% on all rows, because about 1.4% of labels are corrupted and cannot be predicted. Two findings "
        f"shaped the work: the validation period is entirely in the future, and <font name='Courier'>quote_signal</font> "
        f"is a target leak that stops working in that period.", body)]

    s += [Paragraph("1. How I split the data", h2),
          Paragraph("The labelled data runs Jan 1 to Oct 31, 2025. The loads to predict run Nov 1 to Dec 31. So the test that "
                    "matters is forecasting, and every split trains on the past and predicts the next two months, which is the "
                    "same horizon as the real task. A random split was rejected: it mixes future days into training and hides "
                    "time-related failure.", body)]
    rows = [["Train through", "Predict", "Why this fold", "MAPE clean", "MAE clean", "Bias"]]
    why = {"2025-04-30": ("May-Jun", "Same calendar as Nov-Dec: unseen quarter end"),
           "2025-06-30": ("Jul-Aug", "New quarter; quote_signal dead in Aug"),
           "2025-07-31": ("Aug-Sep", "Same calendar as Nov-Dec: unseen quarter end"),
           "2025-08-31": ("Sep-Oct", "Most recent data")}
    for cut, (target, reason) in why.items():
        f = final.loc[cut]
        rows.append([pd.Timestamp(cut).strftime("%b %d"), target, reason, f"{f.mape_clean:.2f}%", f"${f.mae_clean:.0f}", f"{f.bias_clean_pct:+.2f}%"])
    s += [table(rows, [24, 18, 76, 20, 19, 17]), Spacer(1, 4),
          Paragraph("Bias is mean log(actual / predicted). Corrupted rows are removed inside each fold's training window only; "
                    "cleaning never sees future labels. For the final fit, all clean rows through Oct 31 are used, since the "
                    "folds have already chosen the design.", small)]

    s += [Paragraph("2. Why quote_signal was removed", h2),
          Paragraph("In Jan-Mar, Jun and Sep this field is the rate per mile plus about 1% noise. In Apr, May, Jul and Oct it is "
                    "inverted (roughly 4.15 minus rate per mile). In August it is noise, and it is noise in validation too, "
                    "which can be checked without labels by correlating it with a proxy built from distance and equipment.", body),
          picture(quote_signal_figure()),
          table([["Experiment", "MAPE clean"],
                 ["Random 80/20 split, with quote_signal: looks excellent", f"{trap.mape_clean['random 80/20 split, with quote_signal']:.2f}%"],
                 ["August held out, with quote_signal: the honest result", f"{trap.mape_clean['August held out, with quote_signal']:.2f}%"],
                 ["August held out, without quote_signal", f"{trap.mape_clean['August held out, without quote_signal']:.2f}%"]], [150, 24]),
          Spacer(1, 3),
          Paragraph("These three runs use one plain LightGBM. A Sep-Oct holdout alone would not catch the problem, because the "
                    "field is live then and appears to help.", small)]

    s += [Paragraph("3. Model and comparison", h2),
          Paragraph(f"Trees cannot extrapolate past the last training date, so time is handled by a linear model on log rate per mile: "
                    f"market index (elasticity {coef['log_market_index']:.2f}), a linear trend ({coef['days_since_start'] * 30:.1%} per 30 days) and "
                    f"a ramp over the last 31 days of each quarter that resets on day one. The ramp differs by equipment, and its "
                    f"size held steady across all three observed quarter ends: Dry Van +{coef['ramp_dry_van']:.1%}, Reefer +{coef['ramp_reefer']:.1%}, "
                    f"Flatbed +{coef['ramp_flatbed']:.1%}. LightGBM with L1 loss then learns the lane and load shape from "
                    f"distance, weight, equipment and coordinates, on the target with time removed. A small shrunken offset per lane finishes it.", body),
          picture(time_figure())]
    rows = [["Model (mean of 4 folds)", "MAPE clean", "MAE clean", "MAPE all rows", "RMSE all rows"]]
    for name in summary.index:
        r = summary.loc[name]
        rows.append([name, f"{r.mape_clean:.2f}%", f"${r.mae_clean:.0f}", f"{r.mape_all:.2f}%", f"${r.rmse_all:.0f}"])
    s += [table(rows, [74, 24, 24, 26, 26]), Spacer(1, 3),
          Paragraph("All-row RMSE is dominated by corrupted labels, which is why it barely separates the models. These scores are "
                    "slightly optimistic, because the same folds chose the feature design and settings.", small)]

    s += [Paragraph("4. Data quality", h2),
          table([["Issue", "Size", "Action"],
                 ["Corrupted rates, about 3.5x and 0.3x true value", "677 train rows (1.4%)", "Dropped from training by model residual over 0.20 in log terms"],
                 ["Negative weights", "292 train, 145 validation", "Absolute value; residuals match positive weights"],
                 ["Missing weight", "300 train, 165 validation", "Train median plus a missing flag"],
                 ["Missing market index", "374 train, 249 validation", "That date's mean, sd within a day is about 0.025"],
                 ["Capped values", "Weight 5,000 to 47,500; distance floor 70", "Kept as is"],
                 ["Coordinates distorted (Boston at -69.5 lon, Los Angeles at 28.6 lat)", "All cities", "Consistent per city, so used as location codes only"],
                 ["Eight cities only in validation", "12.1% of validation rows", "No city IDs in LightGBM; coordinates carry location"]], [76, 44, 54]),
          Spacer(1, 3),
          Paragraph(f"To test unseen cities I hid eight from training and predicted Sep-Oct: {cold.mape_clean['rows touching a hidden city']:.2f}% MAPE on "
                    f"their rows against {cold.mape_clean['all other rows']:.2f}% on the rest.", body)]

    s += [KeepTogether([Paragraph("5. Fixed December prediction chart", h2),
          picture(ROOT / "scorer_results" / "candidate_december.png"),
          Paragraph(f"The chart inputs have no market index, so each day uses the mean market index of that date's validation loads. "
                    f"The weekly wave comes from the index, the climb from the trend and the Dry Van ramp.", body)])]

    s += [Paragraph("6. Assumptions and risks", h2),
          Paragraph(f"<b>Holidays:</b> labelled data has no Thanksgiving or Christmas. Jul 4 and Memorial Day show no effect, so "
                    f"I modelled none, and the chart shows no holiday dip or spike. <b>December ramp:</b> it rests on three past quarter "
                    f"ends. Removing it would take Dec 31 from ${last.with_ramp:.0f} to ${last.without_ramp:.0f} on the chart lane. "
                    f"<b>Trend:</b> it is extrapolated two months past the data, and fold bias reached {final.bias_clean_pct.abs().max():.1f}%. "
                    f"<b>Market index:</b> using it assumes it is known on the day of the quote; if it is not, the index would need "
                    f"forecasting in production.", body)]

    SimpleDocTemplate(str(HERE / "freight_rate_report.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=15 * mm, title="Freight Rate Prediction: Validation Report",
                      author="Harshit").build(s)


if __name__ == "__main__":
    build()
