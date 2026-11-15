"""Build forks_presentation.pptx: plain default-Office slides, explanation in the notes.
Run: uv run --with python-pptx --with pillow python presentation/build_deck.py
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"
OUTPUTS = HERE.parent / "outputs"

prs = Presentation()  # the stock Office template: white, Calibri
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = 13.333, 7.5
L_TITLE, L_CONTENT, L_TITLE_ONLY = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[5]
MARGIN, BODY_TOP = 0.6, 1.55


def place(shape, x, y, w, h):
    shape.left, shape.top, shape.width, shape.height = Inches(x), Inches(y), Inches(w), Inches(h)


def title(slide, text):
    t = slide.shapes.title
    place(t, MARGIN, 0.3, SW - 2 * MARGIN, 1.1)
    t.text = text
    t.text_frame.paragraphs[0].font.size = Pt(36)


def bullets(body, items, size=20):
    """items: str or (str, level)."""
    tf = body.text_frame
    tf.clear()
    for i, it in enumerate(items):
        text, level = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.font.size = Pt(size if level == 0 else size - 3)
        p.space_after = Pt(6)


def picture(slide, path, x, y, max_w, max_h):
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - pw) / 2), Inches(y + (max_h - ph) / 2),
                             Inches(pw), Inches(ph))


def caption(slide, text, x, y, w, size=12):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.4))
    tb.text_frame.word_wrap = True
    tb.text_frame.text = text
    tb.text_frame.paragraphs[0].font.size = Pt(size)
    tb.text_frame.paragraphs[0].font.italic = True


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = "\n\n".join(p.strip() for p in text.strip().split("\n\n"))


def bullet_slide(t, items, note, size=20):
    s = prs.slides.add_slide(L_CONTENT)
    title(s, t)
    body = s.placeholders[1]
    place(body, MARGIN, BODY_TOP, SW - 2 * MARGIN, SH - BODY_TOP - 0.5)
    bullets(body, items, size)
    notes(s, note)
    return s


def split_slide(t, items, img, note, text_w=5.2, size=18, cap=None):
    """Bullets left, picture right."""
    s = prs.slides.add_slide(L_CONTENT)
    title(s, t)
    body = s.placeholders[1]
    place(body, MARGIN, BODY_TOP, text_w, SH - BODY_TOP - 0.5)
    bullets(body, items, size)
    px = MARGIN + text_w + 0.3
    pw = SW - px - MARGIN
    picture(s, img, px, BODY_TOP, pw, SH - BODY_TOP - (0.9 if cap else 0.5))
    if cap:
        caption(s, cap, px, SH - 0.85, pw)
    notes(s, note)
    return s


def picture_slide(t, img, note, cap=None, items=None, size=18):
    """Picture across the full width, optional short bullets below it."""
    s = prs.slides.add_slide(L_TITLE_ONLY)
    title(s, t)
    bottom = SH - 0.4
    if items:
        tb = s.shapes.add_textbox(Inches(MARGIN), Inches(SH - 1.75), Inches(SW - 2 * MARGIN), Inches(1.4))
        tb.text_frame.word_wrap = True
        for i, text in enumerate(items):
            p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
            p.text = "•  " + text
            p.font.size = Pt(size)
            p.space_after = Pt(4)
        bottom = SH - 1.85
    if cap:
        caption(s, cap, MARGIN, bottom - 0.4, SW - 2 * MARGIN)
        bottom -= 0.45
    picture(s, img, MARGIN, BODY_TOP - 0.1, SW - 2 * MARGIN, bottom - BODY_TOP + 0.1)
    notes(s, note)
    return s


# ------------------------------------------------------------------ 1. title
s = prs.slides.add_slide(L_TITLE)
place(s.shapes.title, 1.0, 2.0, SW - 2.0, 1.6)
s.shapes.title.text = "When do virtual try-on diffusion models decide what?"
place(s.placeholders[1], 1.0, 3.8, SW - 2.0, 1.5)
s.placeholders[1].text = "Comparing the denoising process of CatVTON, OOTDiffusion and IDM-VTON\nMichiel Vanzeir, Honours research, KU Leuven"
notes(s, """
Goal of this meeting: show what I measured, how, and what came out. Three parts: (1) a comparison of the denoising trajectories between models, (2) a within-model analysis of which spatial scales change when, and (3) a causal "fork" experiment that measures when properties are actually decided rather than when they merely look stable.

Everything was run on the ESAT HTCondor GPU cluster (IDM-VTON needs a 24 GB GPU). 48 garment/person pairs x 3 models for the trajectories, 16 pairs x 3 models x 28 samples for the fork experiment.
""")

# ------------------------------------------------------------- 2. question
bullet_slide("Research question", [
    "Diffusion try-on models build the image over ~50 denoising steps",
    "Question: in what order are the garment's properties fixed?",
    ("Pose, color, coarse structure, fine texture, print pattern", 1),
    "Do different try-on architectures follow different processes?",
    ("Coarse-to-fine or everything at once? Separate stages?", 1),
    "Why it matters: explains failures (e.g. lost prints) and shows where guidance or editing can intervene",
], """
Virtual try-on (VTON): given a photo of a person and a photo of a garment, generate the person wearing the garment. The state of the art uses latent diffusion: start from noise and denoise in 50 steps.

At every step the model also implicitly predicts the final clean image (called x0-hat). By looking at this prediction step by step we can see WHEN each property of the garment appears and stops changing.

The question is whether the three models differ in this process: does one fix the coarse shape first and fill in detail later, does another do everything in parallel, are there distinct phases? That tells us something about how the architecture uses the garment information, and it indicates at which step you would have to intervene to change a given property (useful for guidance, editing, or debugging failures such as prints that get lost).
""")

# ---------------------------------------------------------------- 3. models
bullet_slide("Three ways of using the garment", [
    "CatVTON (SD 1.5 inpainting, 512x384)",
    ("Garment and person concatenated into one image; the same UNet processes both at every step", 1),
    "OOTDiffusion-HD (SD 1.5, 1024x768)",
    ("Separate 'outfitting' UNet encodes the clean garment ONCE (t = 0); its features are injected at every step", 1),
    "IDM-VTON (SDXL inpainting, 1024x768)",
    ("Garment UNet re-encodes the garment at EVERY timestep, plus IP-Adapter image prompt and a text description", 1),
    "Same sampler for all: DDIM, 50 steps, eta = 1, seed 42",
], """
The three models were chosen because they inject the garment in structurally different ways.

CatVTON: the simplest. The garment image is concatenated next to the masked person image in latent space, and one Stable-Diffusion-1.5 inpainting UNet denoises the whole thing. No separate garment encoder, no text.

OOTDiffusion: has a second UNet (outfitting UNet) that encodes the clean garment image. I checked in the code: it runs once, at timestep 0, before the denoising loop, and those features are fused into the denoising UNet's self-attention at every step. So the garment information it receives is noise-level independent and identical at every step.

IDM-VTON: SDXL-based and the biggest. Its garment UNet is called inside the loop with the current timestep t, so garment features are recomputed per step. It also uses an IP-Adapter (image prompt) and a short text description of the garment.

To make the comparison fair I used the same sampler everywhere: DDIM with 50 steps and eta = 1 (eta = 1 means fresh noise is injected at every step, which is needed for the fork experiment later), and the same seed. Each model gets the inputs its own official VITON-HD test script uses (masks, DensePose, etc.).
""", size=19)

# ------------------------------------------------------------------ 4. data
split_slide("What we record: x\u0302\u2080 at every step", [
    "VITON-HD test set, unpaired setting",
    "48 hand-picked garments in 8 categories",
    ("solid, lettering, graphic, stripe, floral, dot, check, structural", 1),
    "Every step: decode x̂₀, the model's current guess of the final image",
    "144 trajectories x 50 frames",
], IMG / "x0_strip.png", """
Dataset: VITON-HD test split, the standard benchmark. Unpaired setting = the person wears a different garment than the one we try on (the realistic case).

I hand-picked 6 garments per category (tags in the dataset were too noisy) so we can average per garment type. The categories go from easy (solid colors) to hard (lettering, dense prints, structural garments with bows/collars).

At every denoising step the scheduler gives x0-hat: the model's current estimate of the clean final image. I decode that through the VAE and save it. I verified at step 0 that the scheduler's x0-hat equals the manual formula (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar).

The picture shows one example (a dotted blouse). You can already see differences: OOTDiffusion has the dots from step ~10, CatVTON only fills them in near the end, and IDM-VTON starts very noisy (its first-step prediction is nearly pure noise) and then shows the bow from step ~20.
""", text_w=4.6, size=18, cap="Example: x̂₀ at selected steps for one pair (dot_01)")

# --------------------------------------------------------------- 5. metrics
bullet_slide("Measuring 'how far along' each frame is", [
    "Every frame compared to its own final image, score in [0, 1], 1 = identical",
    "Same resolution and the same garment mask for all models",
    "Axes:",
    ("Pose: OpenPose keypoint agreement (PCK)", 1),
    ("Color: share of garment pixels within ΔE 2.5-20 of the final color (Lab, smoothed)", 1),
    ("Structure: SSIM on blurred grayscale (coarse layout)", 1),
    ("Texture: amount of fine detail (mid-high frequency energy)", 1),
    ("Pattern: similarity of the radial frequency spectrum (print scale)", 1),
    "Summary per curve: area under the curve (AUC); high = model commits early",
], """
Each axis turns one frame into a number between 0 and 1: how close is this frame to the run's own final image on that property. Higher is always "closer to final".

To keep the models comparable, all frames are resized to the same analysis resolution and every garment score uses the same pixel mask (the dataset's agnostic mask of the upper-body clothing region).

Pose: OpenPose keypoints (the same detector the models are conditioned on), percentage of keypoints within a tolerance of the final skeleton.
Color: Lab color difference Delta-E, scored continuously from 1 at 2.5 (just noticeable) down to 0 at 20 (clearly another color), on slightly smoothed images so it measures region color, not individual pixels.
Structure: SSIM on blurred grayscale, i.e. coarse layout/folds/shape.
Texture: how much fine-detail energy there is relative to the final image.
Pattern: compares the radial frequency spectrum, i.e. whether the print has the right scale/period.

Summary: the area under each curve. A high AUC means the frame was close to its final state for most of the trajectory, i.e. the model committed early.
""", size=19)

# -------------------------------------------------------------- 6. validation
split_slide("Does each axis measure its own property?", [
    "Perturb the 144 final images in a controlled way: blur, color shift, warp, noise",
    "Each axis should react to its 'own' perturbation only",
    "Result: color, texture and structure are specific; none react to noise",
    "Limits: pattern overlaps with texture; structure reacts only weakly to warping",
], OUTPUTS / "_analysis" / "axis_validation" / "specificity.png", """
Before trusting the curves I validated the axes. I took every final image, applied a known perturbation (blur removes detail, a color shift changes color, a small warp changes structure, noise should change nothing meaningful) and measured how much each axis drops.

A good axis reacts strongly to its own perturbation (bold cells, the diagonal) and not to the others.

The first version of the metrics failed this test: color only reacted above Delta-E 8 (like an on/off switch), and texture counted noise as detail. That also explained a weird start in the IDM curves. I fixed both (a continuous color score on smoothed images; a frequency band that excludes the highest, noise-dominated frequencies) and all results in this presentation use the fixed version.

Remaining honest limitations: pattern also reacts to blur (so it is not independent of texture), and structure reacts only weakly to warping (max 0.14).
""", text_w=5.6, size=18)

# ---------------------------------------------------------- 7. result 1 chart
by_model = {}
import csv  # noqa: E402
for r in csv.DictReader(open(OUTPUTS / "_comparison" / "by_model.csv")):
    by_model[r["model"]] = r
axes_ = ["color", "structure", "texture", "pattern"]
cd = CategoryChartData()
cd.categories = [a.capitalize() for a in axes_]
for m in ["CatVTON", "OOTDiffusion", "IDM-VTON"]:
    cd.add_series(m, [round(float(by_model[m][f"{a}_auc_mean"]), 2) for a in axes_])
s = prs.slides.add_slide(L_TITLE_ONLY)
title(s, "Result 1: OOTDiffusion commits earliest")
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MARGIN), Inches(BODY_TOP), Inches(7.4), Inches(5.4), cd)
ch = gf.chart
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.value_axis.minimum_scale = 0.6
ch.value_axis.maximum_scale = 1.0
ch.value_axis.has_title = True
ch.value_axis.axis_title.text_frame.text = "AUC (mean over 48 pairs)"
ch.plots[0].has_data_labels = True
ch.plots[0].data_labels.font.size = Pt(10)
ch.plots[0].data_labels.number_format = "0.00"
ch.plots[0].data_labels.number_format_is_linked = False
tb = s.shapes.add_textbox(Inches(8.4), Inches(BODY_TOP + 0.2), Inches(4.3), Inches(5))
tb.text_frame.word_wrap = True
for i, t in enumerate([
    "OOTD higher than both others in 77-98% of pairs, on every axis (p < 10⁻⁴)",
    "CatVTON and IDM-VTON are similar",
    "Pose is fixed from the first steps in all models (not informative)",
    "Caveat: 'close to its own final image' is correlational",
]):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.text = "•  " + t
    p.font.size = Pt(18)
    p.space_after = Pt(10)
notes(s, """
Main cross-model result from the trajectories: OOTDiffusion's frames are closer to their final state earlier than the other two models, on all four garment axes. Paired over the same 48 garments (Wilcoxon signed-rank):
- color: OOTD higher than CatVTON in 94% of pairs, than IDM in 98% (p ~ 1e-11)
- structure: 85% / 94%
- texture: 77% / 79% (p ~ 1e-5)
- pattern: 83% / 88%
CatVTON and IDM-VTON are close to each other.

Pose is not shown: all three models have the pose fixed from the first steps (PCK ~0.98), because the unmasked body pins the skeleton. So pose says nothing about differences.

Important caveat (and the motivation for the next two parts): these curves measure similarity to the run's OWN final image. A model that simply corrects itself less along the way would also look "early". So this shows that OOTD becomes stable earlier, not yet that it DECIDES earlier. Also, comparing axes WITHIN one model is not valid here, because the axes start at different baseline levels (e.g. pattern starts at ~0.7 by construction).
""")

# ------------------------------------------------------- 8. decomposition
picture_slide("Result 2: coarse-to-fine, but differently", IMG / "bands_top.png", """
Second analysis, within each model and without thresholds. For every step I take the change between consecutive x0-hat frames and split it into spatial frequency bands: coarse (overall shape and color regions), mid (pattern-scale structures) and fine (edges, small details). The plot shows, per band, how much of the distance to the final image has already been covered.

Findings (48 pairs each):
- All three models are coarse-to-fine in 100% of pairs (p ~ 1e-15). That is a general property of diffusion models, as expected.
- OOTDiffusion: the three bands progress almost in parallel; its detail phase starts around step 7, versus step ~17-18 for the other two (p < 1e-5). It has the lowest coarse-to-fine index in all 8 categories.
- CatVTON: the most staged. Coarse shape comes quickly, fine detail is mostly filled in during the last ~10 steps. It also fixes chroma (color) before luminance (light/shadow) in 79% of pairs (p = 2e-5); the other two do both together.
- IDM-VTON: after a noisy start, a lot of change stays in the mid band (pattern scale) for a long period, roughly steps 10-40, a kind of 'working phase'.

Caveat: the fine band jumping to 1 at the end is partly by definition (fine detail can only exactly equal the final image at the end). The reference-free measure (which share of each step's change is in which band) confirms the same ordering independently.
""", cap="Progress of each spatial-frequency band towards the final image (mean over 48 pairs, band = interquartile range)", items=[
    "Coarse-to-fine in 100% of pairs for all three models (diffusion in general)",
    "OOTD: bands progress in parallel, detail from step ~7 (others: ~17-18)",
    "CatVTON: most staged, detail in the last ~10 steps  |  IDM: long mid-band 'working phase'",
], size=16)

# ------------------------------------------------------ 9. fork method
split_slide("Result 3: forks \u2013 when is it decided?", [
    "Run up to step k with seed 42, then continue with 4 different noise seeds",
    "What all 4 forks still share was already decided at step k",
    "k = 0, 5, 10, 15, 20, 30, 40; k = 0 = fully independent baseline",
    "16 pairs (2 per category) x 3 models x 28 samples",
    "decided(k): 0 = as different as independent samples, 1 = identical",
    "Causal, same procedure for every model, each axis scaled to its own baseline",
], IMG / "forks_sheet_catvton.png", """
This is the causal experiment. The problem with the previous two analyses is that they compare to the final image, so they measure when a frame LOOKS stable, not when the outcome is actually determined.

Procedure: run the denoising with seed 42 up to step k. Then split into 4 continuations that each get different random noise (with DDIM eta = 1 the scheduler injects fresh noise at every step, so that is the only thing that differs). If all four continuations end up with the same color, the color was already decided at step k. If they end up different, it was still open.

k = 0 means the forks also start from different initial noise: fully independent samples, the 'nothing decided' baseline.

For each axis I compute the agreement between all pairs of fork final images (same metrics as before), and rescale: decided(k) = (agreement(k) - agreement(0)) / (1 - agreement(0)). So 0 = no more agreement than independent samples, 1 = identical. Because each axis is scaled to its own baseline and ceiling, axes CAN now be compared within one model, which fixes the limitation of result 1.

Technical check I did: I verified that forks that branch after the last step are bit-identical, i.e. the shared prefix really is shared (this caught a bug where CatVTON's VAE encoder sampled from the global random generator).

The picture: CatVTON, one pair, rows = fork step, columns = the 4 seeds. At k=0 the garment design itself differs (bow or no bow); at k=40 the forks are nearly the same.
""", text_w=6.3, size=17, cap="CatVTON, dot_01: rows = k, columns = seeds")

# ---------------------------------------------------- 10. fork result models
s = picture_slide("Forks: OOTD decides ~10 steps earlier",
                  IMG / "forks_color_structure.png", """
This is the key result. Curves: decided(k) averaged over 16 pairs, band = standard error. Dashed line = half decided.

Half-decision step (median over pairs):
- Structure: OOTD ~10, IDM ~22, CatVTON ~25
- Color: OOTD ~19, IDM ~29, CatVTON ~32
OOTD is earlier than both others in 94-100% of pairs (paired Wilcoxon p < 1e-3).
IDM decides structure ~3 steps before CatVTON (p = 0.02); for color the difference is not significant (p = 0.07).

Interpretation:
1. This confirms result 1 causally. OOTD does not just look stable earlier; its outcome is actually determined earlier. That rules out the 'it just corrects itself less' explanation.
2. CatVTON and IDM decide almost nothing in the first 10 steps (decided ~0 up to k = 10), whereas OOTD is already ~30% decided on structure at k = 5.
3. Possible explanation (hypothesis, not tested): OOTD extracts garment features once from the clean garment at t = 0 and injects the same features at every step, so strong, noise-free garment information is available from step 1. CatVTON and IDM process the garment together with the current noise level at every step.
""", cap="decided(k), mean ± s.e.m. over 16 pairs; dashed = half decided", items=[
    "Half decided (median step): structure OOTD 10 | IDM 22 | CatVTON 25;  color OOTD 19 | IDM 29 | CatVTON 32",
    "OOTD earlier than both in 94-100% of pairs (p < 10⁻³); CatVTON and IDM decide ~nothing before step 10",
], size=17)

# --------------------------------------------------- 11. within model
split_slide("Forks: structure before color", [
    "In all three models structure is decided 6-8 steps before color (p < 0.001 each)",
    "Even forking at step 40 (last 20%): color only 70-89% decided",
    "CatVTON: garment design (bow, neckline, drape) still varies at k = 30",
    "OOTD's forks agree much earlier (right: same pair, OOTD)",
], IMG / "forks_sheet_ootd.png", """
Within-model ordering, now valid because every axis is scaled to its own baseline:
- In all three models the coarse structure/layout is decided before the color: median gap 6-8 steps, p < 0.001 in each model. So the process is: first where things are, then what color they are.
- (Note: in result 2 CatVTON changed chroma before luminance. That is a different question: which kind of change happens, versus when the outcome gets fixed. Both can be true.)

Late freedom: even forking at step 40 of 50, the color is only 70% (CatVTON), 76% (IDM) or 89% (OOTD) decided. So a considerable part of the result is still being chosen in the last steps.

The contact sheet on this slide is OOTDiffusion on the same pair as the CatVTON sheet on the previous slide: its forks already look alike from k = 20. In CatVTON, even the garment's design (whether there is a bow, the neckline, the drape) differs between forks up to k = 30. The model 'hallucinates' garment structure and commits to it late.
""", text_w=6.3, size=18, cap="OOTDiffusion, dot_01: rows = k, columns = seeds")

# ---------------------------------------------------- 12. limitations
bullet_slide("Limitations", [
    "Texture and pattern are not usable in the fork experiment",
    ("Independent samples already agree ~0.92-0.98: too little headroom, noisy curves", 1),
    "Forks: 16 pairs, 4 seeds, one shared prefix (seed 42) per pair",
    "Hand-designed metrics; validated, but pattern overlaps with texture",
    "The architectural explanation for OOTD is a hypothesis, not tested",
    "One sampler setting (DDIM, 50 steps, eta = 1)",
], """
Be upfront about these:

1. Texture/pattern in forks: two completely independent samples already have almost the same AMOUNT of fine detail and the same spectrum (agreement 0.92-0.98), so there is almost no room to measure a rise. 40 of 48 texture curves are non-monotone; some even go negative. So I only draw conclusions from color and structure in the fork experiment. A fix would be a detail measure that compares the fine frequency band of two forks pixel by pixel instead of only its amount; that can be computed from the existing images, no new cluster runs needed.

2. Sample size: 16 pairs x 4 seeds per fork point. Effects on color/structure are large and consistent (94-100% of pairs), but category-level claims would need more pairs.

3. Metrics are hand-designed; I validated them with perturbations but they are not perfect.

4. The OOTD explanation (clean garment features extracted once) is plausible given the code, but I haven't tested it; one could test it by e.g. extracting OOTD's garment features at the current noise level instead.

5. Only one sampler configuration; eta = 1 is needed for the forks, but the models are normally run with other schedulers/settings.
""", size=20)

# ---------------------------------------------------- 13. conclusions
bullet_slide("Conclusions and next steps", [
    "The three denoising processes clearly differ",
    ("OOTDiffusion decides early and in parallel across scales", 1),
    ("CatVTON is the most staged: shape first, detail at the very end", 1),
    ("IDM-VTON has a long mid-scale 'working phase'", 1),
    "All: coarse-to-fine, structure decided before color, much still open late",
    "Next steps",
    ("Better detail measure for forks (pixel-wise fine-band comparison)", 1),
    ("Test the OOTD hypothesis: garment features at the current noise level", 1),
    ("More pairs per category; other samplers", 1),
], """
Summary in one sentence: yes, the diffusion processes differ clearly between models, and the fork experiment shows this is a real difference in when things are decided, not an artefact of how we measure stability.

- OOTDiffusion: decides structure and color ~10 steps earlier than the others, and its scales progress in parallel.
- CatVTON: most sequential; coarse first, fine detail in the last ~10 steps; garment design stays open long.
- IDM-VTON: similar timing to CatVTON, slightly earlier structure, with a long phase where mostly mid-scale (pattern-scale) content changes.
- Shared: coarse-to-fine, structure before color, and still 10-30% of color freedom left at step 40.

Questions I would like input on:
- Is this enough for a paper (e.g. a workshop paper), or should I first do the next steps?
- Which next step is most valuable: the better detail measure, the causal test of the OOTD hypothesis, or scaling up?
""", size=20)

out = HERE / "forks_presentation.pptx"
prs.save(out)
print(out)
