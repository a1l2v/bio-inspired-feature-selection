// Builds presentation/Bio_Inspired_Feature_Selection.pptx (SEE presentation deck).
// Style follows the user's reference deck (light grey background, Outfit + Instrument Sans, lavender accent,
// section pill, paper-wave imagery); the cover reproduces the reference cover (Untitled presentation.svg):
// off-white panel, pink and violet pills, and the wave image at 50 % opacity with diagonally rounded corners.
// Structure comes from whitespace, thin rules and numbers instead of boxes.
// Every number comes from the project outputs (Derma/README.md, docs/project_writeup.md).
//
// Usage (packages pptxgenjs, react, react-dom, react-icons, sharp on NODE_PATH; PPTX_SKILL = pptx skill directory):
//   .venv\Scripts\python.exe presentation\prepare_images.py     (once: builds the blended wave PNGs)
//   node presentation/build_deck.js

const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts", "apply_theme.js"));

const OUT = path.join(__dirname, "Bio_Inspired_Feature_Selection.pptx");
const ASSETS = path.join(__dirname, "assets");

const THEME = {
  name: "Lavender Paper",
  headFontFace: "Outfit",
  bodyFontFace: "Instrument Sans",
  colors: {
    dk1: "3D3D3D", lt1: "F0F0F0", dk2: "6B6B6B", lt2: "FFFCFA",
    accent1: "B8A7FF", accent2: "7DDA92", accent3: "FE5418", accent4: "5237E6",
    accent5: "039366", accent6: "E992D5", hlink: "5237E6", folHlink: "039366",
  },
};
const HEX = THEME.colors;
const RULE = "C9C9C9"; // thin divider lines

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625", as in the reference deck
pres.title = "Bio-Inspired Feature Selection for Medical Image Classification";
pres.subject = "Nature-inspired algorithms for feature selection on MedMNIST";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const INK = C.text1, MUTED = C.text2, LAV = C.accent1, MINT = C.accent2, ORANGE = C.accent3, VIOLET = C.accent4;
const H = THEME.headFontFace;

// Content frame: full width below the title (the section pill sits above the title, not in a side column).
const X0 = 0.5, CW = 9.0;

// ---------- layouts ----------
// Cover and closing slide, measured from the reference cover SVG (960 x 540 px = 10" x 5.625").
pres.defineSlideMaster({
  title: "COVER",
  background: { color: C.background1 },
  objects: [
    { rect: { x: 0.63, y: 0.31, w: 7.1, h: 5.0, fill: { color: C.background2 }, line: { color: C.background2 } } },
    // Layout objects only take rect/line/text/image, so the two pills are empty text shapes with rounded geometry.
    { text: { text: "", options: { shape: pres.shapes.ROUNDED_RECTANGLE, x: 0.31, y: 0.31, w: 0.32, h: 5.0, rectRadius: 0.16,
      fill: { color: C.accent6 }, line: { color: C.accent6 } } } },
    { text: { text: "", options: { shape: pres.shapes.ROUNDED_RECTANGLE, x: 7.73, y: 0.31, w: 1.96, h: 0.32, rectRadius: 0.16,
      fill: { color: VIOLET }, line: { color: VIOLET } } } },
    { image: { x: 7.73, y: 0.63, w: 1.96, h: 4.68, path: path.join(ASSETS, "cover_wave.png"), altText: "Abstract white paper waves" } },
    { placeholder: { options: { name: "title", type: "title", x: 1.1, y: 0.64, w: 5.9, h: 1.8, fontSize: 32,
      color: INK, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.1, y: 2.55, w: 5.6, h: 0.7, fontSize: 13,
      color: MUTED, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.3, 0.5, 0.45, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.56, w: CW, h: 0.45, fontSize: 22,
      color: INK, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.2, fontSize: 8, color: MUTED, align: "right", margin: 0 },
});

// ---------- helpers ----------
const iconCache = {};
async function icon(name, color = HEX.dk1) {
  const key = name + color;
  if (!iconCache[key]) {
    const Comp = fa[name];
    if (!Comp) throw new Error("unknown icon " + name);
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: "256" }));
    const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
    iconCache[key] = "image/png;base64," + buf.toString("base64");
  }
  return iconCache[key];
}

async function iconDot(slide, name, x, y, d, fill = LAV, id = name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `dot-${id}` });
  slide.addImage({ data: await icon(name), x: x + d * 0.27, y: y + d * 0.27, w: d * 0.46, h: d * 0.46,
    altText: name.replace(/^Fa/, "") + " icon", objectName: `icon-${id}` });
}

function dot(slide, x, y, d = 0.13, fill = LAV, id = "dot") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: id });
}

function hRule(slide, x, y, w, id = "rule") {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: RULE, width: 0.75 }, objectName: id });
}

function vRule(slide, x, y, h, id = "vrule") {
  slide.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: RULE, width: 0.75 }, objectName: id });
}

function text(slide, runs, opts) {
  slide.addText(runs, { isTextBox: true, margin: 0, valign: "top", color: INK, ...opts });
}

// The reference deck's outlined section pill, kept small and placed above the title.
function pill(slide, label) {
  const w = 0.3 + label.length * 0.075;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0, y: 0.22, w, h: 0.25, rectRadius: 0.125,
    fill: { color: C.background1 }, line: { color: INK, width: 0.75 }, objectName: "section-pill" });
  text(slide, label, { x: X0, y: 0.22, w, h: 0.25, fontSize: 8, align: "center", valign: "middle" });
}

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  pill(s, section);
  return s;
}

// Column x positions for n equal columns across the content frame, with gap g.
function cols(n, g = 0.3) {
  const w = (CW - g * (n - 1)) / n;
  return { w, x: (i) => X0 + i * (w + g) };
}

// ---------- slides ----------
async function build() {
  // 1. Cover
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Opening" });
  s.addText("Bio-Inspired Feature Selection for Medical Image Classification", { placeholder: "title" });
  s.addText("When do nature-inspired algorithms improve accuracy? Controlled studies on four MedMNIST datasets",
    { placeholder: "body" });
  text(s, "PRESENTED BY", { x: 1.1, y: 3.6, w: 3.2, h: 0.2, fontSize: 8, color: MUTED, charSpacing: 1 });
  text(s, [
    { text: "Alvin S Thottappilly", options: { fontFace: H } }, { text: "  1BM23CD005", options: { color: MUTED, breakLine: true } },
    { text: "Chinmayee S Bharadwaj", options: { fontFace: H } }, { text: "  1BM23CD012", options: { color: MUTED, breakLine: true } },
    { text: "Taran Thimmaiah", options: { fontFace: H } }, { text: "  1BM23CD065", options: { color: MUTED } },
  ], { x: 1.1, y: 3.85, w: 3.4, h: 0.8, fontSize: 11, paraSpaceAfter: 3 });
  text(s, "GUIDE", { x: 4.75, y: 3.6, w: 2.0, h: 0.2, fontSize: 8, color: MUTED, charSpacing: 1 });
  text(s, "Asst. Prof. Manoj Kumar", { x: 4.75, y: 3.85, w: 2.2, h: 0.3, fontSize: 11, fontFace: H });
  await iconDot(s, "FaDna", 7.08, 4.69, 0.28, LAV, "cover");
  s.addNotes("This project asks a practical question: when do nature-inspired optimisation algorithms actually " +
    "improve a medical image classifier? We tested this under strict, fair conditions on four MedMNIST datasets, " +
    "and found a clear answer: they help with handcrafted radiomics features, but not with deep network features.");

  // 2. Introduction to Optimization Problems
  pres.addSection({ title: "Background" });
  s = content("Introduction to Optimization Problems", "Background");
  const parts = [
    ["Decision variables", "What we can change: which image features to keep"],
    ["Objective function", "What we minimise: 1 − cross-validated accuracy"],
    ["Constraints", "Rules a solution must obey: exactly 50 features"],
  ];
  for (let i = 0; i < parts.length; i++) {
    const y = 1.2 + i * 0.9;
    text(s, "0" + (i + 1), { x: X0, y, w: 0.5, h: 0.35, fontSize: 16, color: VIOLET, fontFace: H });
    text(s, [
      { text: parts[i][0], options: { fontSize: 14, fontFace: H, breakLine: true } },
      { text: parts[i][1], options: { fontSize: 11, color: MUTED } },
    ], { x: X0 + 0.55, y, w: 3.85, h: 0.7 });
    if (i < parts.length - 1) hRule(s, X0, y + 0.8, 4.4, "part-rule" + i);
  }
  text(s, "Goal: find x* = argmin f(x). When the space is huge and f has no gradient, exact methods fail; " +
    "metaheuristics search it approximately.", { x: X0, y: 4.0, w: 4.4, h: 0.8, fontSize: 11 });
  vRule(s, 5.3, 1.2, 3.65, "intro-divider");
  text(s, [{ text: "2" }, { text: "1820", options: { superscript: true } }],
    { x: 5.7, y: 1.1, w: 3.8, h: 0.8, fontSize: 40, fontFace: H, color: VIOLET });
  text(s, "possible feature subsets (about 10⁵⁴⁸) of our 1,820 radiomics features",
    { x: 5.7, y: 1.95, w: 3.7, h: 0.5, fontSize: 11, color: MUTED });
  text(s, [{ text: "≈ 10" }, { text: "98", options: { superscript: true } }],
    { x: 5.7, y: 2.7, w: 3.8, h: 0.7, fontSize: 34, fontFace: H });
  text(s, "ways to choose exactly 50 of them", { x: 5.7, y: 3.42, w: 3.7, h: 0.3, fontSize: 11, color: MUTED });
  text(s, "A billion checks per second would still take far longer than the age of the universe.",
    { x: 5.7, y: 4.1, w: 3.7, h: 0.6, fontSize: 10, italic: true, color: MUTED });
  s.addNotes("An optimization problem has decision variables, an objective and constraints. Feature selection fits " +
    "this exactly: the variables are which features to keep, the objective is classification error, and the " +
    "constraint is the number of features. With 1,820 features there are 2 to the power 1,820 subsets, and even " +
    "choosing exactly 50 gives about 10 to the 98 options, so exhaustive search is impossible.");

  // 3. What are Nature-Inspired Algorithms?
  s = content("What are Nature-Inspired Algorithms?", "Background");
  text(s, "Population-based metaheuristics that copy problem-solving strategies from nature, such as evolution, " +
    "flocking and hunting, to search huge spaces without needing gradients.",
    { x: X0, y: 1.12, w: CW, h: 0.55, fontSize: 13 });
  const steps = [
    ["Initialise", "Random population of candidate solutions"],
    ["Evaluate", "Score each candidate with the fitness function"],
    ["Select", "Favour fitter candidates, drop weaker ones"],
    ["Update", "Crossover, mutation or swarm moves"],
    ["Stop", "Budget used: return the best solution"],
  ];
  const c5 = cols(5, 0.25);
  hRule(s, X0 + 0.06, 2.06, CW - 0.12, "loop-line");
  for (let i = 0; i < steps.length; i++) {
    const x = c5.x(i);
    dot(s, x, 2.0, 0.13, i === 4 ? VIOLET : LAV, "loop-dot" + i);
    text(s, "0" + (i + 1), { x, y: 2.25, w: c5.w, h: 0.22, fontSize: 11, color: VIOLET, fontFace: H });
    text(s, [
      { text: steps[i][0], options: { fontSize: 14, fontFace: H, breakLine: true } },
      { text: steps[i][1], options: { fontSize: 10.5, color: MUTED } },
    ], { x, y: 2.5, w: c5.w, h: 0.95 });
  }
  const props = [["Gradient-free", "Works with any model, even non-differentiable ones"],
    ["Stochastic", "Each run differs, so every search was repeated 30 times"],
    ["Explore vs exploit", "Search widely first, then refine the best regions"]];
  const c3 = cols(3, 0.4);
  hRule(s, X0, 3.72, CW, "props-rule");
  for (let i = 0; i < props.length; i++) {
    text(s, [
      { text: props[i][0], options: { fontSize: 13, fontFace: H, color: VIOLET, breakLine: true } },
      { text: props[i][1], options: { fontSize: 10.5 } },
    ], { x: c3.x(i), y: 3.92, w: c3.w, h: 0.85 });
  }
  s.addNotes("Nature-inspired algorithms keep a population of candidate solutions and improve it in a loop: " +
    "evaluate, select, update, repeat until the budget runs out. They need no gradients, so they can optimise " +
    "any classifier's accuracy directly. Because they are random, a single run proves nothing, which is why every " +
    "result in this project is averaged over 30 runs.");

  // 4. Classification of Nature-Inspired Algorithms
  s = content("Classification of Nature-Inspired Algorithms", "Background");
  const families = [
    ["FaDna", "Evolutionary", "Natural selection and genetics", "GA, Differential Evolution, Evolution Strategies", "Used: GA", true],
    ["FaDove", "Swarm intelligence", "Collective behaviour of animals", "PSO, Grey Wolf, Whale, Harris Hawks, Competitive Swarm, Ant Colony",
      "Used: PSO, binary PSO, GWO, WOA, HHO, CSO", true],
    ["FaAtom", "Physics- and chemistry-based", "Physical laws", "Simulated Annealing, Gravitational Search", "Not used here", false],
    ["FaUsers", "Human- and society-based", "Human learning and social behaviour", "Teaching–Learning-Based Optimisation, Harmony Search",
      "Not used here", false],
  ];
  vRule(s, 5.0, 1.15, 3.8, "family-v");
  hRule(s, X0, 3.05, CW, "family-h");
  for (let i = 0; i < families.length; i++) {
    const [ic, name, insp, ex, used, isUsed] = families[i];
    const x = X0 + (i % 2) * 4.75, y = 1.15 + Math.floor(i / 2) * 2.05;
    await iconDot(s, ic, x, y + 0.05, 0.42, isUsed ? LAV : C.lt2, "family" + i);
    text(s, [
      { text: name, options: { fontSize: 14, fontFace: H, breakLine: true } },
      { text: "Inspired by " + insp.toLowerCase(), options: { fontSize: 10, italic: true, color: MUTED } },
    ], { x: x + 0.55, y, w: 3.65, h: 0.6 });
    text(s, [
      { text: ex, options: { breakLine: true } },
      { text: used, options: { bold: isUsed, color: isUsed ? VIOLET : MUTED } },
    ], { x: x + 0.55, y: y + 0.68, w: 3.65, h: 1.0, fontSize: 10.5 });
  }
  s.addNotes("Nature-inspired algorithms are usually grouped into four families. This project used one evolutionary " +
    "algorithm, the genetic algorithm, and six swarm-intelligence algorithms: PSO, binary PSO, grey wolf, whale, " +
    "Harris hawks and the competitive swarm optimiser. The final, positive study used GA and CSO.");

  // 5. Problem Statement
  pres.addSection({ title: "Project" });
  s = content("Problem Statement", "Project");
  text(s, "Does choosing image features with a nature-inspired algorithm make a medical image classifier " +
    "significantly more accurate than the same pipeline without it?",
    { x: X0, y: 1.1, w: CW, h: 0.78, fontSize: 16, fontFace: H, color: VIOLET });
  text(s, "Success = a significant gain in test accuracy and AUC over 30 paired runs (Wilcoxon test, Holm-corrected).",
    { x: X0, y: 1.9, w: CW, h: 0.3, fontSize: 10.5, color: MUTED });
  text(s, "COMPARED AGAINST · SAME BUDGET", { x: X0, y: 2.42, w: 4, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  const bases = [["All features", "No selection at all"], ["Random search", "Best of 1,000 random subsets"],
    ["ANOVA filter", "Standard statistical ranking"]];
  for (let i = 0; i < bases.length; i++) {
    const x = X0 + i * 1.75;
    dot(s, x, 2.82, 0.12, LAV, "base-dot" + i);
    text(s, [
      { text: bases[i][0], options: { fontSize: 12, fontFace: H, breakLine: true } },
      { text: bases[i][1], options: { fontSize: 10, color: MUTED } },
    ], { x: x + 0.22, y: 2.74, w: 1.45, h: 0.75 });
  }
  text(s, "FOUR MEDMNIST DATASETS", { x: X0, y: 3.65, w: 4, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  const sets = [["BreastMNIST", "Ultrasound · 780"], ["RetinaMNIST", "Fundus · 1,600"],
    ["PneumoniaMNIST", "Chest X-ray · 5,856"], ["DermaMNIST", "Skin lesions · 10,015 (main study)"]];
  for (let i = 0; i < sets.length; i++) {
    const x = X0 + (i % 2) * 2.5, y = 3.97 + Math.floor(i / 2) * 0.52;
    text(s, [
      { text: sets[i][0], options: { fontSize: 11.5, fontFace: H, color: i === 3 ? VIOLET : INK, breakLine: true } },
      { text: sets[i][1], options: { fontSize: 9.5, color: MUTED } },
    ], { x, y, w: 2.4, h: 0.48 });
  }
  // "With vs without" comparison diagram in the slide's colours (replaces the wave image).
  text(s, "HOW WE COMPARE", { x: 6.05, y: 2.42, w: 3.4, h: 0.25, fontSize: 9, color: MUTED, charSpacing: 1 });
  const node = (label, x, y, w, h, fill, line, color, id) => s.addText(label, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: h / 2, x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.75 },
    fontSize: 9.5, color, align: "center", valign: "middle", margin: 0.03, objectName: id });
  const arrow = (x, y, h, id) => s.addShape(pres.shapes.LINE, { x, y, w: 0, h,
    line: { color: INK, width: 1, endArrowType: "triangle" }, objectName: id });
  node("1,820 radiomics features", 6.4, 2.78, 2.75, 0.38, C.background1, INK, INK, "node-features");
  arrow(6.875, 3.16, 0.32, "arrow-without");
  arrow(8.675, 3.16, 0.32, "arrow-with");
  node([{ text: "Without", options: { fontFace: H, breakLine: true } }, { text: "all · random · filter" }],
    6.05, 3.5, 1.65, 0.55, C.background2, RULE, INK, "node-without");
  node([{ text: "With", options: { fontFace: H, breakLine: true } }, { text: "GA · CSO, 50 features" }],
    7.85, 3.5, 1.65, 0.55, VIOLET, VIOLET, C.background2, "node-with");
  arrow(6.875, 4.05, 0.32, "arrow-without-2");
  arrow(8.675, 4.05, 0.32, "arrow-with-2");
  node("SVM → test accuracy & AUC", 6.4, 4.39, 2.75, 0.38, LAV, LAV, INK, "node-test");
  text(s, "30 paired runs · Wilcoxon test, Holm-corrected", { x: 6.05, y: 4.85, w: 3.45, h: 0.2, fontSize: 8.5,
    italic: true, color: MUTED, align: "center" });
  s.addNotes("The question is whether a nature-inspired algorithm, used to choose features, makes the classifier " +
    "significantly better. 'Without it' is tested three ways: using all features, a random search with the same " +
    "budget, and a standard ANOVA filter. We used four MedMNIST datasets; DermaMNIST, with 10,015 skin-lesion " +
    "images, is the main study.");

  // 6. Implementation and Results (pipeline)
  pres.addSection({ title: "Results" });
  s = content("Implementation and Results", "Results");
  text(s, "Pipeline for the main study (DermaMNIST)", { x: X0, y: 1.1, w: CW, h: 0.3, fontSize: 12, color: VIOLET });
  const pipe = [
    ["Medical images", "Skin lesions, 128 × 128 colour, 7 classes, 10,015 images"],
    ["Feature extraction", "1,820 radiomics features: texture, shape, colour, wavelets"],
    ["Feature selection", "GA or CSO keeps exactly 50; 1,000 evaluations per run"],
    ["Classifier", "RBF SVM on the GPU; repeated with KNN"],
    ["Evaluation", "2,005 unseen test images, 30 independent runs"],
  ];
  hRule(s, X0 + 0.06, 1.66, CW - 0.12, "pipe-line");
  for (let i = 0; i < pipe.length; i++) {
    const x = c5.x(i);
    dot(s, x, 1.6, 0.13, i === 2 ? VIOLET : LAV, "pipe-dot" + i);
    text(s, "0" + (i + 1), { x, y: 1.84, w: c5.w, h: 0.22, fontSize: 11, color: VIOLET, fontFace: H });
    text(s, [
      { text: pipe[i][0], options: { fontSize: 13, fontFace: H, breakLine: true } },
      { text: pipe[i][1], options: { fontSize: 10, color: MUTED } },
    ], { x, y: 2.1, w: c5.w, h: 1.2 });
  }
  hRule(s, X0, 3.52, CW, "tiles-rule");
  const tiles = [["1,820 → 50", "features kept: 97% fewer"], ["30 runs", "per method, paired seeds, 1,000 evaluations each"],
    ["1,560", "search runs across all four datasets"]];
  for (let i = 0; i < tiles.length; i++) {
    text(s, [
      { text: tiles[i][0], options: { fontSize: 24, fontFace: H, breakLine: true } },
      { text: tiles[i][1], options: { fontSize: 10, color: MUTED } },
    ], { x: c3.x(i), y: 3.7, w: c3.w, h: 0.9 });
    if (i < tiles.length - 1) vRule(s, c3.x(i) + c3.w + 0.2, 3.75, 0.75, "tile-v" + i);
  }
  text(s, "Python · PyTorch (GPU least-squares SVM) · scikit-learn · scikit-image · mealpy",
    { x: X0, y: 4.8, w: CW, h: 0.25, fontSize: 9, italic: true, color: MUTED });
  s.addNotes("Each image becomes 1,820 handcrafted radiomics features describing texture, shape, colour and " +
    "frequency. GA or CSO then chooses exactly 50 of them, scoring each candidate set by cross-validated accuracy " +
    "on 3,000 training images. The final SVM is trained on the selected features and tested once on 2,005 unseen " +
    "images. Each method was run 30 times with paired seeds.");

  // 7. Implementation and Results (results)
  s = content("Implementation and Results", "Results");
  text(s, "DermaMNIST · 50 features · SVM · mean of 30 runs on the test set",
    { x: X0, y: 1.1, w: CW, h: 0.3, fontSize: 12, color: VIOLET });
  s.addChart(pres.charts.BAR, [
    { name: "GA", labels: ["Accuracy vs random", "Accuracy vs filter", "AUC vs random", "AUC vs filter"], values: [1.38, 1.88, 1.16, 0.83] },
    { name: "CSO", labels: ["Accuracy vs random", "Accuracy vs filter", "AUC vs random", "AUC vs filter"], values: [1.27, 1.77, 1.04, 0.71] },
  ], {
    x: 0.4, y: 1.45, w: 5.7, h: 3.25, barDir: "col", barGrouping: "clustered", barGapWidthPct: 70,
    chartColors: [HEX.accent4, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    dataLabelFontSize: 9, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1,
    showTitle: true, title: "Gain over the controls (percentage points; AUC × 100)", titleFontSize: 10,
    titleFontFace: "+mn-lt", titleColor: HEX.dk1,
    catAxisLabelFontSize: 9, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2,
    valAxisLabelFontSize: 9, valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: HEX.dk2,
    valAxisMinVal: 0, valAxisMaxVal: 2.4, valAxisMajorUnit: 0.6, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "D6D6D6", size: 0.5 }, catGridLine: { style: "none" },
    plotArea: { fill: { color: HEX.lt1 } },
    showLegend: true, legendPos: "b", legendFontSize: 9, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
    altText: "Clustered bar chart: GA gains 1.38 and 1.88 points accuracy and 1.16 and 0.83 points AUC over random search and the filter; CSO gains 1.27, 1.77, 1.04 and 0.71.",
    objectName: "gain-chart",
  });
  vRule(s, 6.35, 1.5, 3.2, "results-divider");
  const stats = [["78.3%", "GA test accuracy, vs 77.0% random search and 76.5% filter"],
    ["0.936", "GA test AUC, vs 0.925 random search and 0.928 filter"],
    ["27 of 30", "runs where GA beat random search on accuracy (30 of 30 vs the filter)"]];
  for (let i = 0; i < stats.length; i++) {
    text(s, [
      { text: stats[i][0], options: { fontSize: 26, fontFace: H, color: i === 0 ? VIOLET : INK, breakLine: true } },
      { text: stats[i][1], options: { fontSize: 10, color: MUTED } },
    ], { x: 6.65, y: 1.45 + i * 1.1, w: 2.85, h: 1.0 });
  }
  text(s, "Deep CNN features (BreastMNIST, RetinaMNIST): no significant gain in 6 controlled studies, because they were " +
    "too redundant. All 1,820 features reach 80.4%; the gain is at a fixed 50-feature budget.",
    { x: X0, y: 4.78, w: CW, h: 0.35, fontSize: 8.5, italic: true, color: MUTED });
  s.addNotes("With 50 features, GA reached 78.3 percent test accuracy against 77.0 for random search and 76.5 for " +
    "the filter: gains of 1.4 and 1.9 percentage points. AUC rose from 0.925 to 0.936. GA beat random search in " +
    "27 of 30 runs and the filter in all 30. CSO performed the same as GA. On deep CNN features, six earlier " +
    "studies showed no gain, which is part of the finding: selection helps only when features are not redundant.");

  // 8. Performance Metrics and Benchmarking
  s = content("Performance Metrics and Benchmarking", "Results");
  const metrics = [["Accuracy", "Share of test images classified correctly"], ["AUC", "Ranking quality; 0.5 = chance, 1 = perfect"],
    ["Macro-F1", "Precision and recall, every class weighted equally"], ["Balanced accuracy", "Mean recall per class; fair under imbalance"]];
  for (let i = 0; i < metrics.length; i++) {
    const y = 1.15 + i * 0.66;
    text(s, [
      { text: metrics[i][0], options: { fontSize: 12.5, fontFace: H, breakLine: true } },
      { text: metrics[i][1], options: { fontSize: 9.5, color: MUTED } },
    ], { x: X0, y, w: 3.1, h: 0.56 });
    hRule(s, X0, y + 0.6, 3.1, "metric-rule" + i);
  }
  text(s, [
    { text: "Fair benchmarking: ", options: { color: VIOLET } },
    { text: "same 1,000-evaluation budget and same 30 seeds for every method; test set used once; Wilcoxon test, Holm-corrected." },
  ], { x: X0, y: 3.95, w: 3.1, h: 0.95, fontSize: 9.5 });
  const rule = { type: "solid", pt: 0.75, color: RULE }, none = { type: "none" };
  const hdr = (t) => ({ text: t, options: { color: MUTED, fontSize: 9, align: "center", border: [none, none, rule, none] } });
  const cell = (t, first = false) => ({ text: t, options: { color: first ? VIOLET : INK, align: first ? "left" : "center",
    fontFace: first ? H : THEME.bodyFontFace, fontSize: first ? 10.5 : 10, border: [none, none, rule, none] } });
  const rows = [
    [hdr("Comparison"), hdr("SVM\naccuracy"), hdr("SVM\nAUC"), hdr("KNN\naccuracy"), hdr("KNN\nAUC")],
    [cell("GA vs random search", true), cell("+1.38 pp\n(+1.8%)"), cell("+0.012\n(+1.3%)"), cell("+1.47 pp\n(+2.1%)"), cell("+0.018\n(+2.3%)")],
    [cell("GA vs filter", true), cell("+1.88 pp\n(+2.5%)"), cell("+0.008\n(+0.9%)"), cell("+1.85 pp\n(+2.6%)"), cell("+0.007\n(+0.9%)")],
    [cell("CSO vs random search", true), cell("+1.27 pp\n(+1.7%)"), cell("+0.010\n(+1.1%)"), cell("+1.33 pp\n(+1.9%)"), cell("+0.020\n(+2.5%)")],
    [cell("CSO vs filter", true), cell("+1.77 pp\n(+2.3%)"), cell("+0.007\n(+0.8%)"), cell("+1.71 pp\n(+2.4%)"), cell("+0.009\n(+1.1%)")],
  ];
  s.addTable(rows, { x: 3.9, y: 1.15, w: 5.6, colW: [1.6, 1.0, 1.0, 1.0, 1.0], rowH: 0.56,
    fontFace: THEME.bodyFontFace, valign: "middle", margin: 0.04, objectName: "gains-table" });
  text(s, [
    { text: "All 16 gains are significant ", options: { color: VIOLET } },
    { text: "(SVM Holm p ≤ 0.00001; KNN ≤ 0.035). pp = percentage points; relative gain in brackets.", options: { color: MUTED } },
  ], { x: 3.9, y: 4.2, w: 5.6, h: 0.5, fontSize: 9.5 });
  s.addNotes("We report accuracy and AUC as the main metrics, plus macro-F1 and balanced accuracy because the " +
    "classes are imbalanced. Accuracy gains are in percentage points, with the relative gain in brackets; AUC gains " +
    "are raw differences. Every method had the same budget and the same seeds, the test set was used once, and the " +
    "p-values are corrected for multiple comparisons. All 16 gains are significant, with both SVM and KNN.");

  // 9. Advantages and Limitations
  pres.addSection({ title: "Discussion" });
  s = content("Advantages and Limitations", "Discussion");
  const advLim = [
    ["FaCheck", "Advantages", MINT, [
      ["Compact model", "50 of 1,820 features (97% fewer), with significantly higher accuracy than random or filter selection of the same size"],
      ["Robust across classifiers", "The gain appears with both SVM and KNN, so it is not a quirk of one model"],
      ["Gradient-free", "Optimises any classifier's accuracy directly, even when the pipeline is not differentiable"],
      ["Parallel by design", "Many candidate subsets can be scored at once, so the search runs efficiently on a GPU"]]],
    ["FaExclamation", "Limitations", ORANGE, [
      ["No gain on deep CNN features", "In 6 controlled studies (1,470 search runs), selection did not beat random search or all features"],
      ["Small accuracy cost", "50 selected features stay about 2 points below all 1,820 features with the SVM"],
      ["Compute-heavy", "Every run trains 1,000 classifiers to score candidate feature subsets"],
      ["Limited scope", "Positive result on one dataset, with default algorithm settings"]]],
  ];
  vRule(s, 5.0, 1.15, 3.95, "advlim-divider");
  for (let i = 0; i < advLim.length; i++) {
    const [ic, head, col, items] = advLim[i];
    const x = X0 + i * 4.75;
    await iconDot(s, ic, x, 1.12, 0.4, col, "advlim" + i);
    text(s, head, { x: x + 0.52, y: 1.17, w: 3.0, h: 0.32, fontSize: 15, fontFace: H, valign: "middle" });
    for (let j = 0; j < items.length; j++) {
      const y = 1.75 + j * 0.84;
      dot(s, x, y + 0.07, 0.11, col, `advlim-dot${i}-${j}`);
      text(s, [
        { text: items[j][0], options: { fontSize: 12.5, fontFace: H, breakLine: true } },
        { text: items[j][1], options: { fontSize: 10.5, color: MUTED } },
      ], { x: x + 0.25, y, w: 3.95, h: 0.76 });
    }
  }
  s.addNotes("The main advantage is a compact model: 50 features instead of 1,820, chosen significantly better than " +
    "by random search or a standard filter, with two different classifiers. The limitations are honest ones: no " +
    "gain on deep CNN features, a small accuracy cost against using every feature, and high compute per run.");

  // 10. Future Scope
  s = content("Future Scope", "Discussion");
  const future = [
    ["Multi-objective selection", "NSGA-II to trade accuracy against the number of features automatically"],
    ["Richer radiomics", "Full pyradiomics set (GLRLM, GLSZM) on 224 × 224 images"],
    ["More datasets", "Repeat the 50-feature study on other MedMNIST and clinical data"],
    ["Hybrid filter + wrapper", "Pre-filter to a few hundred features, then search: faster"],
    ["Algorithm tuning", "Tune population size, mutation and budget on a separate split"],
    ["Clinical interpretability", "Ask dermatologists whether the selected features make sense"],
  ];
  hRule(s, X0, 3.02, CW, "future-rule");
  for (let i = 0; i < future.length; i++) {
    const x = c3.x(i % 3), y = 1.15 + Math.floor(i / 3) * 2.0;
    text(s, "0" + (i + 1), { x, y, w: 1, h: 0.4, fontSize: 20, fontFace: H, color: VIOLET });
    text(s, [
      { text: future[i][0], options: { fontSize: 13, fontFace: H, breakLine: true } },
      { text: future[i][1], options: { fontSize: 10.5, color: MUTED } },
    ], { x, y: y + 0.48, w: c3.w, h: 1.15 });
  }
  s.addNotes("Natural next steps: a multi-objective search that balances accuracy and feature count, a richer " +
    "radiomics feature set at higher resolution, more datasets, a faster hybrid of filter and search, tuning the " +
    "algorithms' settings, and checking with clinicians whether the selected features are meaningful.");

  // 11. References
  pres.addSection({ title: "References" });
  s = content("References", "References");
  const refs = [
    "Yang J. et al. MedMNIST v2. Scientific Data 10, 41 (2023).",
    "Tschandl P. et al. The HAM10000 dataset. Scientific Data 5, 180161 (2018).",
    "Holland J.H. Adaptation in Natural and Artificial Systems. University of Michigan Press (1975).",
    "Kennedy J., Eberhart R. Particle swarm optimization. Proc. ICNN (1995).",
    "Mirjalili S. et al. Grey wolf optimizer. Advances in Engineering Software 69, 46–61 (2014).",
    "Mirjalili S., Lewis A. The whale optimization algorithm. Advances in Engineering Software 95, 51–67 (2016).",
    "Heidari A.A. et al. Harris hawks optimization. Future Generation Computer Systems 97, 849–872 (2019).",
    "Cheng R., Jin Y. A competitive swarm optimizer for large scale optimization. IEEE Trans. Cybernetics 45, 191–204 (2015).",
    "Haralick R.M. et al. Textural features for image classification. IEEE Trans. SMC 3, 610–621 (1973).",
    "Suykens J.A.K., Vandewalle J. Least squares support vector machine classifiers. Neural Processing Letters 9, 293–300 (1999).",
    "Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 65–70 (1979).",
    "Van Thieu N., Mirjalili S. MEALPY: an open-source library of meta-heuristic algorithms in Python. J. Systems Architecture (2023).",
  ];
  vRule(s, 5.0, 1.15, 3.8, "refs-divider");
  for (let c = 0; c < 2; c++) {
    const part = refs.slice(c * 6, c * 6 + 6);
    text(s, part.map((t, j) => ({ text: t, options: { bullet: { type: "number", numberStartAt: c * 6 + 1 },
      breakLine: j < part.length - 1 } })),
    { x: X0 + c * 4.75, y: 1.15, w: 4.1, h: 3.85, fontSize: 10, paraSpaceAfter: 6 });
  }
  s.addNotes("Key references: the MedMNIST benchmark and HAM10000 dataset, the original papers for each " +
    "algorithm used, the texture features, the least-squares SVM, the Holm correction and the mealpy library.");

  // 12. Thank you (same composition as the cover)
  s = pres.addSlide({ masterName: "COVER", sectionTitle: "References" });
  s.addText("Thank you!", { placeholder: "title" });
  s.addText("Questions and discussion", { placeholder: "body" });
  s.addNotes("Thank the panel and invite questions. Key line to repeat: at a fixed 50-feature budget, GA and CSO " +
    "beat random search and a standard filter significantly, on handcrafted radiomics features.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("saved " + OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
