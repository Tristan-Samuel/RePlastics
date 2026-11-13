const log = document.querySelector("#log");
const logBody = document.querySelector("#log-body");
const logToggle = document.querySelector("#log-toggle");
let seen = -1;
let sawRunning = document.body.dataset.running === "yes";
let wasRunning = sawRunning;
let stickToBottom = true;

function appendLine(line) {
  if (!log) return;
  stickToBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
  log.textContent += (log.textContent ? "\n" : "") + line;
  if (stickToBottom) log.scrollTop = log.scrollHeight;
}

function ensureConfirm(form) {
  let field = form.querySelector('input[name="confirm"]');
  if (!field) {
    field = document.createElement("input");
    field.type = "hidden";
    field.name = "confirm";
    form.appendChild(field);
  }
  field.value = "yes";
}

function setLogOpen(open) {
  if (!logBody || !logToggle) return;
  logBody.hidden = !open;
  logToggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function setBusy(running) {
  document.querySelectorAll(
    'form[action$="/jobs/start"] button, button[form="onnx-form"]'
  ).forEach((button) => {
    button.disabled = running;
  });
  const stop = document.querySelector("#stop-job");
  if (stop) stop.disabled = !running;
  const stdin = document.querySelector("#stdin-form");
  if (stdin) stdin.hidden = !running;
}

function updateStatus(state) {
  const status = document.querySelector("#job-status");
  if (status) {
    if (state.running) status.textContent = state.title + " · " + state.status_line;
    else if (state.command) status.textContent = state.status_line || "Finished";
    else status.textContent = "No job yet.";
  }
  if (state.session) {
    const session = document.querySelector("#session");
    session.textContent = state.session.label;
    session.classList.toggle("up", Boolean(state.session.up));
  }
  setBusy(Boolean(state.running));
  if (state.running && !wasRunning) setLogOpen(true);
  wasRunning = Boolean(state.running);
  if (state.running) sawRunning = true;
  if (
    sawRunning &&
    !state.running &&
    document.body.dataset.watchJob === "yes"
  ) {
    window.location.reload();
  }
}

if (logToggle && logBody) {
  logToggle.addEventListener("click", () => {
    setLogOpen(logBody.hidden);
  });
}

if (log) {
  const source = new EventSource("/jobs/stream");
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (typeof data.index === "number") {
      if (data.index <= seen) return;
      seen = data.index;
      appendLine(data.line);
      return;
    }
    if (data.status) updateStatus(data.status);
  };
}

const copy = document.querySelector("#copy-log");
if (copy) {
  copy.addEventListener("click", () => {
    fetch("/jobs/log")
      .then((response) => response.text())
      .then((text) => navigator.clipboard.writeText(text))
      .catch(() => {
        const area = document.createElement("textarea");
        area.value = log ? log.textContent : "";
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
      });
  });
}

document.querySelectorAll("[data-copy-from]").forEach((button) => {
  button.addEventListener("click", () => {
    const node = document.getElementById(button.dataset.copyFrom);
    const text = node ? node.textContent.trim() : "";
    const done = () => {
      const previous = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = previous;
      }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => {
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
        done();
      });
      return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    done();
  });
});

document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    ensureConfirm(form);
  });
});

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function showSelectedModel(form) {
  const panel = document.querySelector("#model-preview");
  const catalogNode = document.querySelector("#model-catalog");
  if (!panel || !catalogNode || !form.elements.model) return;
  const catalog = JSON.parse(catalogNode.textContent);
  const model = catalog.find((item) => item.rel === form.elements.model.value);
  panel.replaceChildren();
  if (!model) {
    const missing = document.createElement("p");
    missing.className = "warn-line";
    missing.textContent = "That checkpoint is not on this computer.";
    panel.appendChild(missing);
    return;
  }
  const title = document.createElement("h2");
  title.textContent = model.title;
  const file = document.createElement("p");
  file.className = "filename";
  file.textContent = model.rel;
  const meta = document.createElement("p");
  meta.className = "meta";
  const bits = [model.mtime_label, model.size_label];
  if (model.default) bits.push("default");
  if (model.frozen) bits.push(`hold < ${model.threshold}`);
  meta.textContent = bits.join(" · ");
  const scores = document.createElement("p");
  scores.className = "scores";
  [
    ["Validation", model.metrics.validation],
    ["Balanced", model.metrics.balanced],
    ["Test", model.metrics.test],
  ].forEach(([label, value]) => {
    const score = document.createElement("span");
    score.className = "score";
    const key = document.createElement("span");
    key.className = "k";
    key.textContent = label;
    score.append(key, ` ${percent(value)}`);
    scores.appendChild(score);
  });
  const role = document.createElement("p");
  role.className = "hint";
  role.textContent = model.role;
  panel.append(title, file, meta, scores, role);
  if (model.warning) {
    const warning = document.createElement("p");
    warning.className = "warn-line";
    warning.textContent = model.warning;
    panel.appendChild(warning);
  }
}

const trainForm = document.querySelector("#train-form");
if (trainForm) {
  const box = document.querySelector("#preview");
  let previewState = {
    ok: true,
    needs_confirm: trainForm.elements.where.value === "colab",
    message: "This starts a Colab GPU session and runs training. Continue?",
  };
  let timer = null;
  const refresh = () => {
    showSelectedModel(trainForm);
    fetch("/jobs/preview", { method: "POST", body: new FormData(trainForm) })
      .then((response) => response.json())
      .then((data) => {
        previewState = data;
        box.textContent = data.ok ? data.display : data.error;
      })
      .catch(() => {
        previewState = { ok: false };
        box.textContent = "Could not build the command.";
      });
  };
  const schedule = () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(refresh, 150);
  };
  trainForm.addEventListener("input", schedule);
  trainForm.addEventListener("change", refresh);
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = JSON.parse(button.dataset.preset);
      Object.entries(preset).forEach(([key, value]) => {
        const field = trainForm.elements[key];
        if (!field) return;
        if (field.type === "checkbox") field.checked = Boolean(value);
        else field.value = value;
      });
      refresh();
    });
  });
  trainForm.addEventListener("submit", (event) => {
    if (previewState.ok === false) {
      event.preventDefault();
      return;
    }
    if (previewState.needs_confirm) {
      const message = previewState.message || "Continue?";
      if (!window.confirm(message)) {
        event.preventDefault();
        return;
      }
      ensureConfirm(trainForm);
    }
  });
  refresh();
}

const onnxForm = document.querySelector("#onnx-form");
if (onnxForm) {
  const command = document.querySelector("#onnx-command");
  const engine = document.querySelector("#engine-command");
  const hint = document.querySelector("#onnx-hint");
  const filesNode = document.querySelector("#onnx-files");
  const files = new Set(filesNode ? JSON.parse(filesNode.textContent) : []);
  const showOnnx = document.querySelector("#show-onnx");
  const showJson = document.querySelector("#show-json");
  const checkForm = document.querySelector("#onnx-check-form");
  let timer = null;

  const onnxFor = (rel) => (rel.endsWith(".pt") ? `${rel.slice(0, -3)}.onnx` : rel);
  const baseName = (rel) => rel.split("/").pop();

  const syncFiles = () => {
    const model = onnxForm.elements.model.value;
    const dataDir = onnxForm.elements.data_dir.value;
    const onnx = onnxFor(model);
    const json = `${onnx}.json`;
    const onnxName = baseName(onnx);
    const stem = onnxName.endsWith(".onnx") ? onnxName.slice(0, -5) : onnxName;
    if (engine) {
      engine.textContent =
        `/usr/src/tensorrt/bin/trtexec --onnx=${onnxName} ` +
        `--saveEngine=/opt/replastics/${stem}.engine --fp16`;
    }
    const onnxReady = files.has(onnx);
    const jsonReady = files.has(json);
    if (showOnnx) {
      showOnnx.hidden = !onnxReady;
      const input = showOnnx.querySelector("input[name=path]");
      if (input) input.value = onnx;
    }
    if (showJson) {
      showJson.hidden = !jsonReady;
      const input = showJson.querySelector("input[name=path]");
      if (input) input.value = json;
    }
    if (checkForm) {
      checkForm.hidden = !onnxReady;
      const modelInput = checkForm.elements.model;
      const dataInput = checkForm.elements.data_dir;
      if (modelInput) modelInput.value = model;
      if (dataInput) dataInput.value = dataDir;
    }
    if (hint) {
      hint.textContent = onnxReady && jsonReady
        ? `${onnx} and ${json}`
        : `Make ONNX writes ${onnx} and ${json}, then scores test photos in ${dataDir}.`;
    }
    const url = new URL(window.location.href);
    url.searchParams.set("model", model);
    url.searchParams.set("data", dataDir);
    window.history.replaceState(null, "", url);
  };

  const refreshOnnx = () => {
    showSelectedModel(onnxForm);
    syncFiles();
    fetch("/jobs/preview", { method: "POST", body: new FormData(onnxForm) })
      .then((response) => response.json())
      .then((data) => {
        if (command) command.textContent = data.ok ? data.display : data.error;
      })
      .catch(() => {
        if (command) command.textContent = "Could not build the command.";
      });
  };
  onnxForm.addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(refreshOnnx, 150);
  });
  onnxForm.addEventListener("change", refreshOnnx);
  refreshOnnx();
}

setBusy(document.body.dataset.running === "yes");

document.querySelectorAll("[data-filters]").forEach((root) => {
  root.querySelectorAll("select").forEach((select) => {
    select.addEventListener("change", () => {
      const klass = root.querySelector("[data-filter-class]").value;
      const resin = root.querySelector("[data-filter-resin]").value;
      const scope = root.parentElement || document;
      scope.querySelectorAll("[data-true]").forEach((el) => {
        const okClass = klass === "any" || el.dataset.true === klass;
        const okResin = resin === "any" || el.dataset.resin === resin;
        el.hidden = !(okClass && okResin);
      });
    });
  });
});

const tabs = document.querySelector("[data-tabs]");
if (tabs) {
  const names = Array.from(tabs.querySelectorAll("[data-tab]")).map(
    (button) => button.dataset.tab
  );
  const showTab = (name) => {
    const chosen = names.includes(name) ? name : tabs.dataset.defaultTab;
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== chosen;
    });
    tabs.querySelectorAll("[data-tab]").forEach((button) => {
      button.setAttribute(
        "aria-selected",
        button.dataset.tab === chosen ? "true" : "false"
      );
    });
    if (location.hash !== "#" + chosen) {
      history.replaceState(null, "", "#" + chosen);
    }
  };
  tabs.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  });
  let hashed = location.hash.replace("#", "");
  if (hashed === "inspect") hashed = "overview";
  showTab(names.includes(hashed) ? hashed : tabs.dataset.defaultTab);
}

const decide = document.querySelector("[data-decide]");
if (decide) {
  const BAND_CAP = 60;
  const NOTES = {
    validation: "Validation is the set the cutoff was fit on.",
    test: "Test was scored once. It did not pick the cutoff.",
    both: "Validation and test together. The cutoff was fit on validation only.",
    excluded: "Photos left out of the splits. True class is metal.",
  };
  const status = decide.querySelector("[data-status]");
  const note = decide.querySelector("[data-split-note]");
  const canvas = decide.querySelector("[data-hist]");
  const thresholdInput = decide.querySelector("[data-threshold]");
  const thresholdNumber = decide.querySelector("[data-threshold-num]");
  const rangeMin = decide.querySelector("[data-range-min]");
  const rangeMax = decide.querySelector("[data-range-max]");
  const classFilter = decide.querySelector("[data-decide-class]");
  const resinFilter = decide.querySelector("[data-decide-resin]");
  const excludedButton = decide.querySelector('[data-split-value="excluded"]');
  let frames = [];
  let split = "validation";
  let temperature = "";
  let dragging = false;
  let loaded = false;
  let shownThreshold = null;

  function snapThreshold(value) {
    const number = Math.min(0.99, Math.max(0.5, Number(value)));
    if (!Number.isFinite(number)) return 0.85;
    return Math.round(number * 100) / 100;
  }

  function snapRange(value) {
    const number = Math.min(1, Math.max(0.5, Number(value)));
    if (!Number.isFinite(number)) return 0.5;
    return Math.round(number * 100) / 100;
  }

  function fileUrl(path) {
    return "/files/" + path.split("/").map(encodeURIComponent).join("/");
  }

  function percent(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return (value * 100).toFixed(2) + "%";
  }

  function rate(part, whole) {
    if (!whole) return "—";
    return ((100 * part) / whole).toFixed(2) + "%";
  }

  function spread(values) {
    const count = values.length;
    if (!count) return { n: 0, mean: null, std: null };
    const mean = values.reduce((sum, value) => sum + value, 0) / count;
    if (count < 2) return { n: count, mean, std: null };
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
    return { n: count, mean, std: Math.sqrt(variance) };
  }

  function activeFrames() {
    if (split === "both") {
      return frames.filter(
        (frame) => frame.split === "validation" || frame.split === "test"
      );
    }
    return frames.filter((frame) => frame.split === split);
  }

  function passesFilter(frame) {
    const klass = classFilter.value;
    const resin = resinFilter.value;
    if (klass !== "any" && frame.true !== klass) return false;
    if (resin !== "any" && frame.resin !== resin) return false;
    return true;
  }

  function setFigure(name, value, detail, tone) {
    const tile = decide.querySelector(`[data-fig="${name}"]`);
    if (!tile) return;
    tile.querySelector(".num").textContent = value;
    tile.querySelector(".sub").textContent = detail || "";
    tile.classList.remove("bad", "warn", "good", "accent");
    if (tone) tile.classList.add(tone);
  }

  function fillMatrix(rows, threshold) {
    const table = decide.querySelector("[data-matrix]");
    if (!table) return;
    ["metal", "plastic"].forEach((truth) => {
      const counts = { metal: 0, plastic: 0, uncertain: 0 };
      rows.forEach((frame) => {
        if (frame.true !== truth) return;
        if (frame.score < threshold) counts.uncertain += 1;
        else if (frame.pred === "plastic") counts.plastic += 1;
        else counts.metal += 1;
      });
      const line = table.querySelector(`[data-row="${truth}"]`);
      ["metal", "plastic", "uncertain"].forEach((key) => {
        const cell = line.querySelector(`[data-cell="${key}"]`);
        cell.textContent = counts[key].toLocaleString("en-US");
        cell.classList.remove("hit", "miss", "hold", "zero");
        if (!counts[key]) cell.classList.add("zero");
        else if (key === "uncertain") cell.classList.add("hold");
        else if (key === truth) cell.classList.add("hit");
        else cell.classList.add("miss");
      });
    });
  }

  function renderStd(rows) {
    const host = decide.querySelector("[data-std]");
    host.replaceChildren();
    const groups = [
      ["All", rows],
      ["Correct", rows.filter((frame) => frame.pred === frame.true)],
      ["Wrong", rows.filter((frame) => frame.pred !== frame.true)],
    ];
    groups.forEach(([label, group]) => {
      const stats = spread(group.map((frame) => frame.score));
      const tile = document.createElement("div");
      tile.className = "metric";
      const num = document.createElement("span");
      num.className = "num";
      num.textContent =
        stats.mean === null
          ? "—"
          : stats.mean.toFixed(3) +
            (stats.std === null ? "" : " ± " + stats.std.toFixed(3));
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = "n=" + stats.n.toLocaleString("en-US");
      const name = document.createElement("span");
      name.className = "lbl";
      name.textContent = label;
      tile.append(num, sub, name);
      host.appendChild(tile);
    });
  }

  const THUMB_SIZES = ["s", "m", "l"];
  let reviewRows = [];
  let reviewIndex = 0;
  let reviewSplit = false;

  function savedThumbs() {
    try {
      const value = localStorage.getItem("decide-thumbs");
      if (THUMB_SIZES.includes(value)) return value;
    } catch (error) {
      return "s";
    }
    return "s";
  }

  function applyThumbs(size) {
    const next = THUMB_SIZES.includes(size) ? size : "s";
    decide.dataset.thumbs = next;
    decide.querySelectorAll("[data-thumb]").forEach((button) => {
      button.setAttribute(
        "aria-pressed",
        button.dataset.thumb === next ? "true" : "false"
      );
    });
    try {
      localStorage.setItem("decide-thumbs", next);
    } catch (error) {
      /* private mode */
    }
  }

  applyThumbs(savedThumbs());
  decide.querySelectorAll("[data-thumb]").forEach((button) => {
    button.addEventListener("click", () => applyThumbs(button.dataset.thumb));
  });

  function caption(frame, showSplit) {
    const bits = [frame.true + " → " + frame.pred, percent(frame.score)];
    if (frame.resin && frame.resin !== "unknown") bits.push(frame.resin);
    if (showSplit) bits.push(frame.split);
    return bits.join(" · ");
  }

  function kindFor(frame, threshold) {
    const wrong = frame.pred !== frame.true;
    if (frame.score < threshold) return wrong ? "caught" : "held-right";
    return wrong ? "still" : "ok";
  }

  function shortCall(frame) {
    const mark = { metal: "M", plastic: "P" };
    const left = mark[frame.true] || "?";
    const right = mark[frame.pred] || "?";
    return left + "→" + right;
  }

  function paintReview() {
    const viewer = document.querySelector("[data-viewer]");
    const frame = reviewRows[reviewIndex];
    if (!viewer || !frame) return;
    const photoFigure = viewer.querySelector("[data-viewer-photo]");
    const photoImage = photoFigure.querySelector("img");
    const name = frame.path.split("/").pop();
    viewer.querySelector("[data-viewer-title]").textContent = name;
    photoFigure.hidden = false;
    if (frame.image) {
      photoImage.src = fileUrl(frame.image);
      photoImage.alt = name;
    } else {
      photoImage.removeAttribute("src");
      photoImage.alt = "";
    }
    viewer.querySelector("[data-viewer-meta]").textContent = caption(
      frame,
      reviewSplit
    );
    const nav = viewer.querySelector("[data-viewer-nav]");
    const many = reviewRows.length > 1;
    nav.hidden = !many;
    viewer.querySelector("[data-viewer-pos]").textContent =
      reviewIndex + 1 + " of " + reviewRows.length;
    viewer.querySelector("[data-viewer-prev]").disabled = reviewIndex === 0;
    viewer.querySelector("[data-viewer-next]").disabled =
      reviewIndex === reviewRows.length - 1;
    viewer.querySelector("[data-viewer-note]").textContent = many
      ? "Arrow keys move through this group."
      : "";
    setGradcamTarget(frame.image || "");
  }

  function openReview(rows, index, showSplit) {
    const viewer = document.querySelector("[data-viewer]");
    if (!viewer) return;
    reviewRows = rows;
    reviewIndex = index;
    reviewSplit = showSplit;
    const plot = viewer.querySelector("[data-viewer-plot]");
    const hits = viewer.querySelector("[data-viewer-hits]");
    if (plot) plot.hidden = true;
    if (hits) hits.replaceChildren();
    paintReview();
    if (!viewer.open) viewer.showModal();
  }

  function stepReview(delta) {
    const next = reviewIndex + delta;
    if (next < 0 || next >= reviewRows.length) return;
    reviewIndex = next;
    paintReview();
  }

  const viewer = document.querySelector("[data-viewer]");
  if (viewer) {
    viewer.addEventListener("keydown", (event) => {
      const nav = viewer.querySelector("[data-viewer-nav]");
      if (!nav || nav.hidden) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        stepReview(1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        stepReview(-1);
      }
    });
    viewer
      .querySelector("[data-viewer-prev]")
      .addEventListener("click", () => stepReview(-1));
    viewer
      .querySelector("[data-viewer-next]")
      .addEventListener("click", () => stepReview(1));
  }

  function card(frame, showSplit, kind, onOpen) {
    const article = document.createElement("button");
    article.type = "button";
    article.className = frame.image ? "frame" : "frame no-photo";
    if (kind && kind !== "ok") article.dataset.kind = kind;
    const name = frame.path.split("/").pop();
    const summary = caption(frame, showSplit);
    article.title = summary + " · " + name;
    article.setAttribute("aria-label", summary + ", " + name);
    if (frame.image) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.src = fileUrl(frame.image);
      article.appendChild(img);
      const badge = document.createElement("span");
      badge.className = "thumb-score";
      const call = document.createElement("span");
      call.textContent = shortCall(frame);
      const score = document.createElement("span");
      score.textContent = percent(frame.score);
      badge.append(call, score);
      article.appendChild(badge);
    }
    const line = document.createElement("p");
    line.textContent = summary;
    const path = document.createElement("p");
    path.className = "path";
    path.textContent = name;
    article.append(line, path);
    article.addEventListener("click", onOpen);
    return article;
  }

  function fillCards(host, rows, showSplit, kindOf) {
    const cards = host.querySelector("[data-bucket-cards], [data-band-cards]");
    cards.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "None";
      cards.appendChild(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    rows.forEach((frame, index) => {
      const kind = typeof kindOf === "function" ? kindOf(frame) : kindOf;
      fragment.appendChild(
        card(frame, showSplit, kind, () => openReview(rows, index, showSplit))
      );
    });
    cards.appendChild(fragment);
  }

  function render() {
    const rows = activeFrames();
    const threshold = snapThreshold(thresholdInput.value);
    shownThreshold = threshold;
    const viewMin = snapRange(rangeMin.value);
    const viewMax = snapRange(rangeMax.value);
    const wrong = rows.filter((frame) => frame.pred !== frame.true);
    const held = rows.filter((frame) => frame.score < threshold);
    const caught = wrong.filter((frame) => frame.score < threshold);
    const still = wrong.filter((frame) => frame.score >= threshold);
    const heldRight = held.filter((frame) => frame.pred === frame.true);
    const accepted = rows.filter((frame) => frame.score >= threshold);
    const acceptedCorrect = accepted.filter((frame) => frame.pred === frame.true);
    const recalls = ["metal", "plastic"]
      .map((name) => {
        const subset = accepted.filter((frame) => frame.true === name);
        if (!subset.length) return null;
        return (
          subset.filter((frame) => frame.pred === name).length / subset.length
        );
      })
      .filter((value) => value !== null);
    const balanced = recalls.length
      ? recalls.reduce((sum, value) => sum + value, 0) / recalls.length
      : null;
    note.textContent = NOTES[split] + (temperature ? "  T " + temperature : "");
    setFigure(
      "held",
      held.length.toLocaleString("en-US"),
      rate(held.length, rows.length),
      held.length ? "warn" : "good"
    );
    setFigure(
      "caught",
      caught.length + "/" + wrong.length,
      "of the misses",
      caught.length ? "accent" : ""
    );
    setFigure(
      "still",
      still.length.toLocaleString("en-US"),
      "sorted",
      still.length ? "bad" : "good"
    );
    setFigure(
      "held-right",
      heldRight.length.toLocaleString("en-US"),
      "correct",
      heldRight.length ? "warn" : "good"
    );
    setFigure(
      "accuracy",
      rate(acceptedCorrect.length, accepted.length),
      "on the rest",
      "accent"
    );
    setFigure(
      "balanced",
      balanced === null ? "—" : (balanced * 100).toFixed(2) + "%",
      "on the rest",
      "accent"
    );
    renderStd(rows);
    fillMatrix(rows, threshold);
    draw(rows, threshold, viewMin, viewMax);
    const filtered = rows.filter(passesFilter);
    const showSplit = split === "both";
    const byScore = (left, right) => left.score - right.score;
    fillBucket("still", filtered.filter((frame) => frame.pred !== frame.true && frame.score >= threshold).sort(byScore), showSplit);
    fillBucket("caught", filtered.filter((frame) => frame.pred !== frame.true && frame.score < threshold).sort(byScore), showSplit);
    fillBucket("held-right", filtered.filter((frame) => frame.pred === frame.true && frame.score < threshold).sort(byScore), showSplit);
    const band = filtered
      .filter((frame) => frame.score >= viewMin && frame.score <= viewMax)
      .sort(byScore);
    const shown = band.slice(0, BAND_CAP);
    const bandCount = decide.querySelector("[data-band-count]");
    bandCount.textContent =
      band.length > shown.length
        ? "lowest " + shown.length + " of " + band.length.toLocaleString("en-US")
        : band.length.toLocaleString("en-US");
    const bandHost = decide.querySelector("[data-band]");
    fillCards(bandHost, shown, showSplit, (frame) => kindFor(frame, threshold));
  }

  function fillBucket(name, rows, showSplit) {
    const host = decide.querySelector(`[data-bucket="${name}"]`);
    host.querySelector("[data-bucket-count]").textContent = rows.length.toLocaleString("en-US");
    fillCards(host, rows, showSplit, name);
  }

  function draw(rows, threshold, viewMin, viewMax) {
    const width = canvas.clientWidth || 640;
    const height = 260;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const pad = { l: 36, r: 12, t: 12, b: 28 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;
    const span = Math.max(0.01, viewMax - viewMin);
    const xOf = (score) => pad.l + ((score - viewMin) / span) * plotW;
    const binSize = 0.01;
    const binCount = Math.round(0.5 / binSize);
    const correct = new Array(binCount).fill(0);
    const missed = new Array(binCount).fill(0);
    rows.forEach((frame) => {
      let index = Math.floor((frame.score - 0.5) / binSize);
      if (index < 0) index = 0;
      if (index >= binCount) index = binCount - 1;
      if (frame.pred === frame.true) correct[index] += 1;
      else missed[index] += 1;
    });
    let peak = 1;
    for (let index = 0; index < binCount; index += 1) {
      const start = 0.5 + index * binSize;
      if (start + binSize < viewMin || start > viewMax) continue;
      peak = Math.max(peak, correct[index] + missed[index]);
    }
    ctx.fillStyle = "rgba(240, 180, 41, 0.16)";
    const stats = spread(rows.map((frame) => frame.score));
    if (stats.std !== null) {
      const left = Math.max(viewMin, stats.mean - stats.std);
      const right = Math.min(viewMax, stats.mean + stats.std);
      if (right > left) {
        ctx.fillRect(xOf(left), pad.t, xOf(right) - xOf(left), plotH);
      }
      if (stats.mean >= viewMin && stats.mean <= viewMax) {
        ctx.strokeStyle = "#f0b429";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(xOf(stats.mean), pad.t);
        ctx.lineTo(xOf(stats.mean), pad.t + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    for (let index = 0; index < binCount; index += 1) {
      const start = 0.5 + index * binSize;
      const end = start + binSize;
      if (end <= viewMin || start >= viewMax) continue;
      const left = xOf(Math.max(start, viewMin));
      const right = xOf(Math.min(end, viewMax));
      const barW = Math.max(1, right - left - 1);
      const scale = Math.log1p(peak);
      const correctH = (Math.log1p(correct[index]) / scale) * plotH;
      const wrongH = (Math.log1p(missed[index]) / scale) * plotH;
      const base = pad.t + plotH;
      ctx.fillStyle = "#3ee0c5";
      ctx.fillRect(left, base - correctH, barW, correctH);
      if (missed[index]) {
        ctx.fillStyle = "#ff5d73";
        ctx.fillRect(left, base - wrongH, barW, wrongH);
      }
    }
    ctx.strokeStyle = "#243044";
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + plotH);
    ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = "#8b98ab";
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "center";
    const ticks = 5;
    for (let tick = 0; tick < ticks; tick += 1) {
      const score = viewMin + (span * tick) / (ticks - 1);
      const x = xOf(score);
      ctx.fillText(score.toFixed(2), x, height - 8);
    }
    ctx.textAlign = "right";
    ctx.fillText(String(peak), pad.l - 6, pad.t + 10);
    ctx.fillText("log", pad.l - 6, pad.t + plotH / 2);
    ctx.fillText("0", pad.l - 6, pad.t + plotH);
    if (threshold >= viewMin && threshold <= viewMax) {
      ctx.strokeStyle = "#e7edf5";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xOf(threshold), pad.t);
      ctx.lineTo(xOf(threshold), pad.t + plotH);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  function scoreFromPointer(event) {
    const rect = canvas.getBoundingClientRect();
    const padLeft = 36;
    const padRight = 12;
    const plotW = rect.width - padLeft - padRight;
    const ratio = (event.clientX - rect.left - padLeft) / plotW;
    const viewMin = snapRange(rangeMin.value);
    const viewMax = snapRange(rangeMax.value);
    return snapThreshold(viewMin + ratio * (viewMax - viewMin));
  }

  function setThreshold(value) {
    const next = snapThreshold(value);
    thresholdInput.value = String(next);
    thresholdNumber.value = next.toFixed(2);
    if (shownThreshold === next) return;
    render();
  }

  function keepRangeOrdered(changed) {
    let low = snapRange(rangeMin.value);
    let high = snapRange(rangeMax.value);
    if (low > high) {
      if (changed === rangeMin) high = low;
      else low = high;
    }
    if (high - low < 0.01) {
      if (changed === rangeMin) high = Math.min(1, low + 0.01);
      else low = Math.max(0.5, high - 0.01);
    }
    rangeMin.value = String(low);
    rangeMax.value = String(high);
    render();
  }

  thresholdInput.addEventListener("input", () => setThreshold(thresholdInput.value));
  thresholdNumber.addEventListener("change", () => setThreshold(thresholdNumber.value));
  rangeMin.addEventListener("input", () => keepRangeOrdered(rangeMin));
  rangeMax.addEventListener("input", () => keepRangeOrdered(rangeMax));
  classFilter.addEventListener("change", render);
  resinFilter.addEventListener("change", render);
  decide.querySelectorAll("[data-split-value]").forEach((button) => {
    button.addEventListener("click", () => {
      split = button.dataset.splitValue;
      decide.querySelectorAll("[data-split-value]").forEach((item) => {
        item.setAttribute("aria-pressed", item === button ? "true" : "false");
      });
      render();
    });
  });
  const frozen = decide.querySelector("[data-use-frozen]");
  if (frozen) {
    frozen.addEventListener("click", () => setThreshold(decide.dataset.frozen));
  }
  decide.querySelector("[data-tail]").addEventListener("click", () => {
    rangeMin.value = "0.9";
    rangeMax.value = "1";
    render();
  });
  decide.querySelector("[data-full]").addEventListener("click", () => {
    rangeMin.value = "0.5";
    rangeMax.value = "1";
    render();
  });
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    setThreshold(scoreFromPointer(event));
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setThreshold(scoreFromPointer(event));
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (loaded) render();
    }).observe(canvas.parentElement);
  }

  fetch(decide.dataset.scoresUrl)
    .then((response) => {
      if (!response.ok) throw new Error("scores");
      return response.json();
    })
    .then((payload) => {
      frames = payload.frames || [];
      temperature = payload.temperature ? Number(payload.temperature).toFixed(6) : "";
      if (frames.some((frame) => frame.split === "excluded")) {
        excludedButton.hidden = false;
      }
      if (status) status.hidden = true;
      loaded = true;
      const frozenValue = decide.dataset.frozen;
      if (frozenValue) {
        const next = snapThreshold(frozenValue);
        thresholdInput.value = String(next);
        thresholdNumber.value = next.toFixed(2);
      }
      render();
    })
    .catch(() => {
      if (status) status.textContent = "Could not load scores.";
    });
}

const photos = document.querySelector("[data-photos]");
const viewer = document.querySelector("[data-viewer]");
if (photos && viewer) {
  const queryInput = photos.querySelector("[data-photo-query]");
  const splitInput = photos.querySelector("[data-photo-split]");
  const classInput = photos.querySelector("[data-photo-class]");
  const resinInput = photos.querySelector("[data-photo-resin]");
  const originInput = photos.querySelector("[data-photo-origin]");
  const countLine = photos.querySelector("[data-photo-count]");
  const grid = photos.querySelector("[data-photo-grid]");
  const plotFigure = viewer.querySelector("[data-viewer-plot]");
  const plotImage = plotFigure.querySelector("img");
  const photoFigure = viewer.querySelector("[data-viewer-photo]");
  const photoImage = photoFigure.querySelector("img");
  const metaLine = viewer.querySelector("[data-viewer-meta]");
  const noteLine = viewer.querySelector("[data-viewer-note]");
  const hits = viewer.querySelector("[data-viewer-hits]");
  const title = viewer.querySelector("[data-viewer-title]");
  let timer = null;
  let latest = [];

  function fileUrl(path) {
    return "/files/" + path.split("/").map(encodeURIComponent).join("/");
  }

  function focusPhotos() {
    const panel = photos.closest("[data-panel]");
    if (panel && panel.hidden) return;
    photos.scrollIntoView({ block: "nearest" });
  }

  function photoMeta(image) {
    const bits = [];
    if (image.split) bits.push(image.split);
    if (image.class_name) bits.push(image.class_name);
    if (image.resin) bits.push(image.resin);
    else if (image.resin === "") bits.push("no tag");
    if (image.origin_label) bits.push(image.origin_label);
    bits.push(image.name);
    return bits.join(" · ");
  }

  function showPhoto(image) {
    photoFigure.hidden = false;
    photoImage.src = fileUrl(image.rel);
    photoImage.alt = image.name;
    metaLine.textContent = photoMeta(image);
    setGradcamTarget(image.rel || "");
  }

  function openViewer() {
    const nav = viewer.querySelector("[data-viewer-nav]");
    if (nav) nav.hidden = true;
    if (!viewer.open) viewer.showModal();
  }

  function renderHits(images, host, keepPlot) {
    host.replaceChildren();
    images.forEach((image) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "frame photo-hit";
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = image.name;
      img.src = fileUrl(image.rel);
      const caption = document.createElement("p");
      caption.textContent = image.name;
      button.append(img, caption);
      button.addEventListener("click", () => {
        if (!keepPlot) {
          plotFigure.hidden = true;
          noteLine.textContent = "";
          hits.replaceChildren();
        }
        showPhoto(image);
        title.textContent = image.name;
        openViewer();
      });
      host.appendChild(button);
    });
  }

  function search(extra) {
    const params = new URLSearchParams();
    const query = (extra && extra.query !== undefined ? extra.query : queryInput.value).trim();
    if (extra && extra.query !== undefined) queryInput.value = extra.query;
    if (query) params.set("q", query);
    const bare = Boolean(extra && extra.bare);
    if (!bare && splitInput && splitInput.value) params.set("split", splitInput.value);
    if (!bare && classInput && classInput.value) params.set("class", classInput.value);
    if (!bare && resinInput && resinInput.value) params.set("resin", resinInput.value);
    if (!bare && originInput && originInput.value) params.set("origin", originInput.value);
    const idle = !params.toString();
    if (idle && !(extra && extra.force)) {
      countLine.textContent = "Type part of a filename, or set a filter.";
      grid.replaceChildren();
      latest = [];
      return Promise.resolve({ images: [], total: 0 });
    }
    countLine.textContent = "Searching…";
    return fetch(photos.dataset.endpoint + "?" + params.toString())
      .then((response) => {
        if (!response.ok) throw new Error("search");
        return response.json();
      })
      .then((payload) => {
        latest = payload.images || [];
        const total = payload.total || 0;
        countLine.textContent = total
          ? total.toLocaleString("en-US") +
            (total > latest.length ? ", showing " + latest.length : "")
          : "No photos match.";
        renderHits(latest, grid, false);
        return { images: latest, total: total };
      })
      .catch(() => {
        countLine.textContent = "Could not search photos.";
        grid.replaceChildren();
        latest = [];
        return { images: [], total: 0 };
      });
  }

  function schedule() {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => search(), 180);
  }

  queryInput.addEventListener("input", schedule);
  [splitInput, classInput, resinInput, originInput].forEach((input) => {
    if (input) input.addEventListener("change", () => search({ force: true }));
  });

  viewer.querySelector("[data-viewer-close]").addEventListener("click", () => {
    viewer.close();
  });

  document.querySelectorAll(".shot > a").forEach((link) => {
    link.addEventListener("click", (event) => {
      const figure = link.parentElement;
      if (!figure || !figure.dataset.plot) return;
      event.preventDefault();
      plotFigure.hidden = false;
      plotImage.src = link.href;
      plotImage.alt = figure.querySelector("figcaption")
        ? figure.querySelector("figcaption").textContent.trim()
        : "Analysis";
      photoFigure.hidden = true;
      photoImage.removeAttribute("src");
      setGradcamTarget("");
      hits.replaceChildren();
      title.textContent = "Analysis";
      const source = figure.dataset.source;
      if (source) {
        showPhoto({
          rel: source,
          name: figure.dataset.sourceName || source.split("/").pop(),
          split: figure.dataset.sourceSplit || "",
          class_name: figure.dataset.sourceClass || "",
          resin: figure.dataset.sourceResin || "",
          origin_label: figure.dataset.sourceOrigin || "",
        });
        noteLine.textContent = "This plot is that photo.";
        title.textContent = figure.dataset.sourceName || "Photo";
        openViewer();
        focusPhotos();
        return;
      }
      const lookup = figure.dataset.query || "";
      noteLine.textContent = lookup
        ? "Looking up " + lookup + "."
        : "This plot is not tied to one filename.";
      openViewer();
      focusPhotos();
      if (!lookup) return;
      search({ query: lookup, force: true, bare: true }).then((payload) => {
        const images = payload.images || [];
        const total = payload.total || 0;
        if (total === 1 && images.length === 1) {
          showPhoto(images[0]);
          noteLine.textContent = "One photo matches " + lookup + ".";
          title.textContent = images[0].name;
        } else if (images.length) {
          noteLine.textContent =
            total.toLocaleString("en-US") +
            " photos match " +
            lookup +
            (total > images.length ? ". Showing " + images.length + "." : ". Pick one.");
          renderHits(images, hits, true);
        } else {
          noteLine.textContent = "No photo filename matches " + lookup + ".";
        }
      });
    });
  });
}

function setGradcamTarget(path) {
  const form = document.querySelector("[data-gradcam]");
  if (!form) return;
  const field = form.querySelector("[data-gradcam-path]");
  const value = path || "";
  if (field) field.value = value;
  form.hidden = !value;
}

document.querySelectorAll("[data-gradcam-upload]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    const file = form.querySelector('input[type="file"]');
    const path = form.querySelector('input[name="images"]');
    const hasFile = file && file.files && file.files.length > 0;
    const hasPath = path && path.value.trim();
    const note = form.querySelector("[data-gradcam-need]");
    if (hasFile || hasPath) {
      if (note) note.hidden = true;
      return;
    }
    event.preventDefault();
    if (note) note.hidden = false;
  });
});

const MODEL_SORTS = new Set([
  "newest",
  "oldest",
  "name",
  "name-desc",
  "balanced",
  "validation",
  "test",
  "size",
]);

function checkpointName(value) {
  const name = value.trim();
  if (!name) return name;
  if (name.toLowerCase().endsWith(".pt")) return name.slice(0, -3) + ".pt";
  return name + ".pt";
}

function modelNumber(row, key) {
  const raw = row.dataset[key];
  if (raw === undefined || raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function applyModelList() {
  const tools = document.querySelector("[data-model-tools]");
  const list = document.querySelector("[data-model-list]");
  if (!tools || !list) return;
  const filter = tools.querySelector("[data-model-filter]");
  const sort = tools.querySelector("[data-model-sort]");
  const count = tools.querySelector("[data-model-count]");
  const empty = document.querySelector("[data-model-empty]");
  const query = filter.value.trim().toLocaleLowerCase();
  const mode = MODEL_SORTS.has(sort.value) ? sort.value : "newest";
  const rows = [...list.children].map((row, index) => ({ row, index }));
  rows.sort((a, b) => {
    let diff = 0;
    if (mode === "oldest") {
      diff = modelNumber(a.row, "mtime") - modelNumber(b.row, "mtime");
    } else if (mode === "name" || mode === "name-desc") {
      diff = (a.row.dataset.title || "").localeCompare(
        b.row.dataset.title || "",
        undefined,
        { sensitivity: "base", numeric: true }
      );
      if (mode === "name-desc") diff = -diff;
    } else if (mode === "newest") {
      diff = modelNumber(b.row, "mtime") - modelNumber(a.row, "mtime");
    } else {
      const left = modelNumber(a.row, mode);
      const right = modelNumber(b.row, mode);
      if (left === null && right === null) diff = 0;
      else if (left === null) diff = 1;
      else if (right === null) diff = -1;
      else diff = right - left;
    }
    return diff || a.index - b.index;
  });
  let shown = 0;
  rows.forEach(({ row }) => {
    const haystack = [
      row.dataset.title,
      row.dataset.filename,
      row.dataset.role,
      row.dataset.aliases,
      row.dataset.warning,
    ]
      .join(" ")
      .toLocaleLowerCase();
    const match = !query || haystack.includes(query);
    row.hidden = !match;
    if (match) shown += 1;
    list.append(row);
  });
  if (count) {
    count.hidden = !query;
    count.textContent = shown + " of " + rows.length;
  }
  if (empty) empty.hidden = shown !== 0;
}

function rememberModelQuery(form) {
  const tools = document.querySelector("[data-model-tools]");
  if (!tools || !form.elements.q || !form.elements.sort) return;
  const filter = tools.querySelector("[data-model-filter]");
  const sort = tools.querySelector("[data-model-sort]");
  form.elements.q.value = filter ? filter.value : "";
  form.elements.sort.value = sort ? sort.value : "";
}

function writeModelQuery() {
  const tools = document.querySelector("[data-model-tools]");
  if (!tools) return;
  const filter = tools.querySelector("[data-model-filter]");
  const sort = tools.querySelector("[data-model-sort]");
  const params = new URLSearchParams(window.location.search);
  const query = filter.value.trim();
  const mode = sort.value;
  if (query) params.set("q", query);
  else params.delete("q");
  if (mode && mode !== "newest") params.set("sort", mode);
  else params.delete("sort");
  const text = params.toString();
  const next =
    window.location.pathname + (text ? "?" + text : "") + window.location.hash;
  const current =
    window.location.pathname + window.location.search + window.location.hash;
  if (next !== current) window.history.replaceState(null, "", next);
}

const modelTools = document.querySelector("[data-model-tools]");
if (modelTools) {
  const filter = modelTools.querySelector("[data-model-filter]");
  const sort = modelTools.querySelector("[data-model-sort]");
  const params = new URLSearchParams(window.location.search);
  if (filter && params.get("q")) filter.value = params.get("q");
  if (sort && MODEL_SORTS.has(params.get("sort"))) sort.value = params.get("sort");
  applyModelList();
  filter.addEventListener("input", () => {
    applyModelList();
    writeModelQuery();
  });
  sort.addEventListener("change", () => {
    applyModelList();
    writeModelQuery();
  });
}

document.querySelectorAll("form.rename-form").forEach((form) => {
  form.addEventListener("submit", (event) => {
    rememberModelQuery(form);
    const file = form.elements.filename;
    if (!file) return;
    const nextName = checkpointName(file.value);
    const previous = checkpointName(file.defaultValue);
    if (!nextName || nextName === previous) return;
    const message =
      "Rename the file to " +
      nextName +
      "? Links inside models/ that point at it are updated. A command that still uses the old path will miss it.";
    if (!window.confirm(message)) event.preventDefault();
  });
});

const openSet = document.querySelector("[data-open-set]");
if (openSet) {
  const SCORE_LABELS = {
    cloud: "Cloud distance",
    call: "Call distance",
    knn: "k-NN distance",
    maha: "Mahalanobis",
  };
  const SPLIT_NOTES = {
    validation: "Validation is the set the cutoff was fit on.",
    test: "Test was scored once. It did not pick the cutoff.",
    both: "Validation and test together. The cutoff was fit on validation only.",
  };
  const STRIP_CAP = 36;
  const confidenceInput = openSet.querySelector("[data-confidence]");
  const confidenceNumber = openSet.querySelector("[data-confidence-num]");
  const distanceInput = openSet.querySelector("[data-distance]");
  const distanceNumber = openSet.querySelector("[data-distance-num]");
  const scoreSelect = openSet.querySelector("[data-score]");
  const disagreeInput = openSet.querySelector("[data-disagree]");
  const keepInput = openSet.querySelector("[data-keep]");
  const keepValue = openSet.querySelector("[data-keep-value]");
  const keepNumber = openSet.querySelector("[data-keep-num]");
  const fitGoal = openSet.querySelector("[data-fit-goal]");
  const fitWhat = openSet.querySelector("[data-fit-what]");
  const fitNote = openSet.querySelector("[data-fit-note]");
  const keptStrip = openSet.querySelector("[data-kept]");
  const keptCount = openSet.querySelector("[data-kept-count]");
  const canvas = openSet.querySelector("[data-scatter]");
  const status = openSet.querySelector("[data-status]");
  const note = openSet.querySelector("[data-split-note]");
  const rule = openSet.querySelector("[data-rule]");
  const disagreeNote = openSet.querySelector("[data-disagree-note]");
  const strip = openSet.querySelector("[data-strip]");
  const stripCount = openSet.querySelector("[data-strip-count]");
  const probeForm = openSet.querySelector("[data-probe-form]");
  const probeStatus = openSet.querySelector("[data-probe-status]");
  const probeGrid = openSet.querySelector("[data-probes]");
  const dialog = document.querySelector("[data-open-viewer]");
  const dialogImage = dialog.querySelector("[data-open-image]");
  const dialogCaption = dialog.querySelector("[data-open-caption]");
  const dialogTitle = dialog.querySelector("[data-open-title]");
  let payload = null;
  let frames = [];
  let probes = [];
  let split = "validation";
  let scoreKey = "cloud";
  let points = [];
  let viewList = [];
  let viewIndex = 0;
  const HOLD_CAP = 0.05;
  const LIST_STEP = 240;
  const listSources = { probe: true, validation: false, test: false };
  let listFilter = "all";
  let listSort = "problems";
  let listCap = LIST_STEP;
  let zoom = { x0: 0.5, x1: 1, y0: 0, y1: null };
  let rubber = null;
  let pointer = null;
  let hoverKey = "";
  const scatterTip = openSet.querySelector("[data-scatter-tip]");
  const zoomNote = openSet.querySelector("[data-zoom-note]");
  const listMore = openSet.querySelector("[data-list-more]");

  function fileUrl(path) {
    return "/files/" + String(path).split("/").map(encodeURIComponent).join("/");
  }

  function trimNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    return String(Math.round(number * 1e6) / 1e6);
  }

  function snapConfidence(value) {
    const number = Math.min(0.99, Math.max(0.5, Number(value)));
    if (!Number.isFinite(number)) return 0.85;
    return Math.round(number * 100) / 100;
  }

  function snapKeep(value) {
    let number = Number(value);
    if (!Number.isFinite(number)) return 0.95;
    if (number > 1) number /= 100;
    number = Math.min(0.999, Math.max(0.5, number));
    return Math.round(number * 1000) / 1000;
  }

  function formatKeep(value) {
    return snapKeep(value).toFixed(3);
  }

  function formatDistance(value) {
    if (!Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }

  function confidenceCut() {
    return snapConfidence(confidenceInput.value);
  }

  function distanceCut() {
    const number = Number(distanceInput.value);
    if (!Number.isFinite(number) || number < 0) return 0;
    return number;
  }

  function keepCut() {
    return Math.max(confidenceCut(), snapKeep(keepValue.value));
  }

  function currentRule() {
    return {
      keepOn: Boolean(keepInput.checked),
      keep: keepCut(),
      distance: distanceCut(),
      key: scoreKey,
      floor: confidenceCut(),
      clash: disagreeInput.checked,
    };
  }

  function judge(frame, rule) {
    const confidence = Number(frame.confidence);
    const distance = Number(frame[rule.key]);
    const far = Number.isFinite(distance) && distance > rule.distance;
    const clash = rule.clash && frame.disagree;
    const rescued = rule.keepOn && confidence >= rule.keep && (far || clash);
    let decision;
    if (rule.keepOn && confidence >= rule.keep) decision = frame.pred || "accepted";
    else if (far || clash) decision = "others";
    else if (confidence < rule.floor) decision = "uncertain";
    else decision = frame.pred || "accepted";
    return { decision, rescued, confidence };
  }

  function axisMax(key) {
    let max = Number(payload && payload.ranges ? payload.ranges[key] : 1);
    if (!Number.isFinite(max) || max <= 0) max = 1;
    probes.forEach((frame) => {
      const value = Number(frame[key]);
      if (Number.isFinite(value)) max = Math.max(max, value * 1.08);
    });
    return max;
  }

  function visibleFrames() {
    return frames.filter((frame) => {
      if (split === "both") return frame.split === "validation" || frame.split === "test";
      return frame.split === split;
    });
  }

  function decisionFor(frame) {
    return judge(frame, currentRule()).decision;
  }

  function known(frame) {
    return frame.true === "metal" || frame.true === "plastic";
  }

  function mistake(frame) {
    return known(frame) && frame.pred !== frame.true;
  }

  function kindFor(frame) {
    const decision = decisionFor(frame);
    const accepted = decision !== "others" && decision !== "uncertain";
    if (accepted) {
      if (frame.true === "other" || mistake(frame)) return "misidentified";
      return "kept";
    }
    return decision === "uncertain" ? "uncertain" : "others";
  }

  function kindLabel(kind) {
    if (kind === "kept") return "kept";
    if (kind === "others") return "discarded · others";
    if (kind === "uncertain") return "discarded · uncertain";
    return "misidentified";
  }

  function measure(rows, rule, collect) {
    const stats = {
      n: rows.length,
      accepted: 0,
      uncertain: 0,
      others: 0,
      kept: 0,
      known: 0,
      knownOthers: 0,
      missed: 0,
      wrong: 0,
      caught: 0,
      heldRight: 0,
      uncertainRight: 0,
      uncertainWrong: 0,
      acceptedCorrect: 0,
      metalAccepted: 0,
      metalHit: 0,
      plasticAccepted: 0,
      plasticHit: 0,
      clash: 0,
      confidentMistakes: 0,
    };
    const matrix = {
      metal: { metal: 0, plastic: 0, uncertain: 0, others: 0 },
      plastic: { metal: 0, plastic: 0, uncertain: 0, others: 0 },
    };
    const newly = [];
    const keptRows = [];
    rows.forEach((frame) => {
      const call = judge(frame, rule);
      const decision = call.decision;
      if (frame.disagree) stats.clash += 1;
      if (call.rescued) {
        stats.kept += 1;
        if (collect) keptRows.push(frame);
      }
      if (decision === "others") stats.others += 1;
      else if (decision === "uncertain") stats.uncertain += 1;
      else stats.accepted += 1;
      if (!known(frame)) return;
      stats.known += 1;
      const right = frame.pred === frame.true;
      if (!right) stats.wrong += 1;
      if (!right && call.confidence >= rule.floor) stats.confidentMistakes += 1;
      if (decision === "others") {
        stats.knownOthers += 1;
        if (collect && call.confidence >= rule.floor) newly.push(frame);
      }
      if (decision === "uncertain" || decision === "others") {
        if (right) stats.heldRight += 1;
        else stats.caught += 1;
      } else if (right) {
        stats.acceptedCorrect += 1;
      } else {
        stats.missed += 1;
      }
      if (decision === "uncertain") {
        if (right) stats.uncertainRight += 1;
        else stats.uncertainWrong += 1;
      }
      const bucket = decision === "uncertain" || decision === "others"
        ? decision
        : (frame.pred === "plastic" ? "plastic" : "metal");
      matrix[frame.true][bucket] += 1;
      if (decision !== "uncertain" && decision !== "others") {
        if (frame.true === "metal") {
          stats.metalAccepted += 1;
          if (frame.pred === "metal") stats.metalHit += 1;
        } else if (frame.true === "plastic") {
          stats.plasticAccepted += 1;
          if (frame.pred === "plastic") stats.plasticHit += 1;
        }
      }
    });
    const recalls = [];
    if (stats.metalAccepted) recalls.push(stats.metalHit / stats.metalAccepted);
    if (stats.plasticAccepted) recalls.push(stats.plasticHit / stats.plasticAccepted);
    stats.balanced = recalls.length
      ? recalls.reduce((sum, value) => sum + value, 0) / recalls.length
      : null;
    stats.hold = stats.n ? (stats.uncertain + stats.others) / stats.n : 1;
    return { stats, matrix, newly, keptRows };
  }

  function paintList(host, countNode, rows) {
    const shown = rows.slice(0, STRIP_CAP);
    host.replaceChildren();
    shown.forEach((frame) => host.appendChild(card(frame, rows)));
    if (!rows.length) {
      countNode.textContent = "None";
      return;
    }
    countNode.textContent = rows.length > shown.length
      ? `Showing ${shown.length} of ${rows.length}`
      : countLabel(rows.length);
  }

  function fillOpenMatrix(matrix) {
    const table = openSet.querySelector("[data-matrix]");
    if (!table) return;
    ["metal", "plastic"].forEach((truth) => {
      const line = table.querySelector(`[data-row="${truth}"]`);
      ["metal", "plastic", "uncertain", "others"].forEach((key) => {
        const cell = line.querySelector(`[data-cell="${key}"]`);
        const count = matrix[truth][key];
        cell.textContent = countLabel(count);
        cell.classList.remove("hit", "miss", "hold", "zero");
        if (!count) cell.classList.add("zero");
        else if (key === "uncertain" || key === "others") cell.classList.add("hold");
        else if (key === truth) cell.classList.add("hit");
        else cell.classList.add("miss");
      });
    });
  }

  function setFigure(name, value, sub) {
    const node = openSet.querySelector(`[data-fig="${name}"]`);
    if (!node) return;
    node.querySelector(".num").textContent = value;
    const line = node.querySelector(".sub");
    if (line) line.textContent = sub || "";
  }

  function share(part, whole) {
    if (!whole) return "—";
    return ((100 * part) / whole).toFixed(2) + "%";
  }

  function countLabel(value) {
    return Number(value).toLocaleString("en-US");
  }

  function setProbeFigure(name, value, sub) {
    const node = openSet.querySelector(`[data-probe-fig="${name}"]`);
    if (!node) return;
    node.querySelector(".num").textContent = value;
    const line = node.querySelector(".sub");
    if (line) line.textContent = sub || "";
  }

  function card(frame, list) {
    const decision = decisionFor(frame);
    const kind = kindFor(frame);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frame";
    button.dataset.kind = kind;
    const trueName = frame.true || "unlabeled";
    button.title = `${kindLabel(kind)} · true ${trueName} · ${frame.path || ""}`;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = frame.path || "photo";
    img.src = fileUrl(frame.path);
    const overlay = document.createElement("span");
    overlay.className = "thumb-score";
    const left = document.createElement("span");
    const shownCall = decision === "others" || decision === "uncertain"
      ? decision
      : (frame.resin || frame.pred || decision);
    left.textContent = shownCall;
    const right = document.createElement("span");
    right.textContent = formatDistance(Number(frame[scoreKey]));
    overlay.append(left, right);
    button.append(img, overlay);
    const mixed = [listSources.probe, listSources.validation, listSources.test]
      .filter(Boolean).length > 1;
    if (mixed) {
      const tag = document.createElement("span");
      tag.className = "thumb-split";
      tag.textContent = frame.split === "validation"
        ? "val"
        : frame.split === "test"
          ? "test"
          : "probe";
      button.append(tag);
    }
    button.addEventListener("click", () => openFrame(frame, list));
    return button;
  }

  function paintStrip(rows) {
    paintList(strip, stripCount, rows);
  }

  function galleryRows() {
    const rows = [];
    if (listSources.probe) probes.forEach((frame) => rows.push(frame));
    frames.forEach((frame) => {
      if (listSources.validation && frame.split === "validation") rows.push(frame);
      if (listSources.test && frame.split === "test") rows.push(frame);
    });
    return rows;
  }

  function passesListFilter(frame) {
    const kind = kindFor(frame);
    if (listFilter === "kept") return kind === "kept";
    if (listFilter === "discarded") return kind === "others" || kind === "uncertain";
    if (listFilter === "misidentified") return kind === "misidentified";
    return true;
  }

  function compareGallery(a, b) {
    if (listSort === "confidence") return Number(b.confidence) - Number(a.confidence);
    if (listSort === "distance") return Number(b[scoreKey]) - Number(a[scoreKey]);
    if (listSort === "added") return 0;
    const order = { misidentified: 0, others: 1, uncertain: 2, kept: 3 };
    const delta = (order[kindFor(a)] || 0) - (order[kindFor(b)] || 0);
    if (delta) return delta;
    const kind = kindFor(a);
    if (kind === "uncertain") return Number(a.confidence) - Number(b.confidence);
    if (kind === "others") return Number(b[scoreKey]) - Number(a[scoreKey]);
    return Number(b.confidence) - Number(a.confidence);
  }

  function paintProbes() {
    probeGrid.replaceChildren();
    const gallery = galleryRows();
    const counts = {
      accepted: 0,
      uncertain: 0,
      others: 0,
      unknown: 0,
      unknownCaught: 0,
      known: 0,
      knownOthers: 0,
      misidentified: 0,
    };
    gallery.forEach((frame) => {
      const decision = decisionFor(frame);
      if (decision === "others") counts.others += 1;
      else if (decision === "uncertain") counts.uncertain += 1;
      else counts.accepted += 1;
      if (kindFor(frame) === "misidentified") counts.misidentified += 1;
      if (frame.true === "other") {
        counts.unknown += 1;
        if (decision === "others") counts.unknownCaught += 1;
      } else if (known(frame)) {
        counts.known += 1;
        if (decision === "others") counts.knownOthers += 1;
      }
    });
    const sorted = gallery.filter(passesListFilter).slice().sort(compareGallery);
    const shown = sorted.slice(0, listCap);
    shown.forEach((frame) => probeGrid.appendChild(card(frame, sorted)));
    const note = openSet.querySelector("[data-probe-note]");
    if (note) {
      if (!gallery.length) {
        note.textContent = "Score probes, or turn on validation or test.";
      } else if (shown.length < sorted.length) {
        note.textContent = listFilter === "all"
          ? `Showing ${shown.length} of ${sorted.length}. The percentages count every photo in the selected splits.`
          : `Showing ${shown.length} of ${sorted.length} in this group. The percentages still count the whole list.`;
      } else if (listFilter !== "all") {
        note.textContent = `${sorted.length} in this group. The percentages still count the whole list.`;
      } else {
        note.textContent = "";
      }
    }
    if (listMore) {
      const left = sorted.length - shown.length;
      listMore.hidden = left <= 0;
      listMore.textContent = left > 0 ? `Show ${Math.min(LIST_STEP, left)} more` : "Show more";
    }
    const total = gallery.length;
    setProbeFigure(
      "accepted",
      share(counts.accepted, total),
      total ? `${counts.accepted} of ${total}` : ""
    );
    setProbeFigure(
      "uncertain",
      share(counts.uncertain, total),
      total ? `${counts.uncertain} of ${total}` : ""
    );
    setProbeFigure(
      "others",
      share(counts.others, total),
      total ? `${counts.others} of ${total}` : ""
    );
    setProbeFigure(
      "caught",
      share(counts.unknownCaught, counts.unknown),
      !total ? "" : counts.unknown
        ? `${counts.unknownCaught} of ${counts.unknown} unknown`
        : "no unknown label"
    );
    setProbeFigure(
      "known-others",
      share(counts.knownOthers, counts.known),
      !total ? "" : counts.known
        ? `${counts.knownOthers} of ${counts.known} known`
        : "no metal or plastic folder"
    );
    setProbeFigure(
      "misidentified",
      share(counts.misidentified, total),
      total ? `${counts.misidentified} still accepted` : ""
    );
  }

  function fullBounds() {
    return { x0: 0.5, x1: 1, y0: 0, y1: axisMax(scoreKey) };
  }

  function currentBounds() {
    const full = fullBounds();
    return {
      x0: zoom.x0,
      x1: zoom.x1,
      y0: zoom.y0,
      y1: zoom.y1 == null ? full.y1 : zoom.y1,
    };
  }

  function clampBounds(next) {
    const full = fullBounds();
    let x0 = next.x0;
    let x1 = next.x1;
    let y0 = next.y0;
    let y1 = next.y1;
    if (!(x1 > x0) || !(y1 > y0)) return;
    const minX = 0.02;
    const minY = Math.max(full.y1 * 0.02, 1e-4);
    if (x1 - x0 < minX) {
      const mid = (x0 + x1) / 2;
      x0 = mid - minX / 2;
      x1 = mid + minX / 2;
    }
    if (y1 - y0 < minY) {
      const mid = (y0 + y1) / 2;
      y0 = mid - minY / 2;
      y1 = mid + minY / 2;
    }
    const fullX = full.x1 - full.x0;
    const fullY = full.y1 - full.y0;
    if (x1 - x0 >= fullX - 1e-9) {
      x0 = full.x0;
      x1 = full.x1;
    } else if (x0 < full.x0) {
      x1 += full.x0 - x0;
      x0 = full.x0;
      if (x1 > full.x1) x1 = full.x1;
    } else if (x1 > full.x1) {
      x0 -= x1 - full.x1;
      x1 = full.x1;
      if (x0 < full.x0) x0 = full.x0;
    }
    if (y1 - y0 >= fullY - 1e-9) {
      y0 = full.y0;
      y1 = full.y1;
    } else if (y0 < full.y0) {
      y1 += full.y0 - y0;
      y0 = full.y0;
      if (y1 > full.y1) y1 = full.y1;
    } else if (y1 > full.y1) {
      y0 -= y1 - full.y1;
      y1 = full.y1;
      if (y0 < full.y0) y0 = full.y0;
    }
    const covers = Math.abs(x0 - full.x0) < 1e-6
      && Math.abs(x1 - full.x1) < 1e-6
      && Math.abs(y0 - full.y0) < 1e-6
      && Math.abs(y1 - full.y1) < 1e-6;
    zoom.x0 = x0;
    zoom.x1 = x1;
    zoom.y0 = y0;
    zoom.y1 = covers ? null : y1;
  }

  function resetZoom() {
    zoom = { x0: 0.5, x1: 1, y0: 0, y1: null };
  }

  function layout() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const pad = { left: 48, right: 14, top: 14, bottom: 32 };
    const plotW = Math.max(1, width - pad.left - pad.right);
    const plotH = Math.max(1, height - pad.top - pad.bottom);
    const bounds = currentBounds();
    const xSpan = bounds.x1 - bounds.x0 || 1;
    const ySpan = bounds.y1 - bounds.y0 || 1;
    return {
      width,
      height,
      pad,
      plotW,
      plotH,
      bounds,
      xOf: (value) => pad.left + ((value - bounds.x0) / xSpan) * plotW,
      yOf: (value) => pad.top + (1 - (value - bounds.y0) / ySpan) * plotH,
      xData: (px) => bounds.x0 + ((px - pad.left) / plotW) * xSpan,
      yData: (py) => bounds.y1 - ((py - pad.top) / plotH) * ySpan,
    };
  }

  function axisTicks(min, max, count) {
    const span = max - min;
    if (!(span > 0)) return [min];
    const raw = span / count;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const err = raw / pow;
    const nice = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const step = nice * pow;
    const start = Math.ceil((min - step * 1e-8) / step) * step;
    const ticks = [];
    for (let index = 0; index < count + 4; index += 1) {
      const value = start + index * step;
      if (value > max + step * 1e-6) break;
      if (value >= min - step * 1e-6) ticks.push(value);
    }
    return ticks.length ? ticks : [min, max];
  }

  function eventLocal(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function nearestPoint(x, y, limit) {
    let best = null;
    let bestD = limit;
    points.forEach((point) => {
      const distance = Math.hypot(point.x - x, point.y - y);
      if (distance < bestD) {
        best = point;
        bestD = distance;
      }
    });
    let stacked = [];
    if (best) {
      stacked = points.filter((point) => Math.hypot(point.x - best.x, point.y - best.y) <= 8);
    }
    return { best, stacked };
  }

  function placeTip(point, stacked, event) {
    if (!scatterTip) return;
    if (!point) {
      scatterTip.hidden = true;
      return;
    }
    const frame = point.frame;
    const name = (frame.path || "photo").split("/").pop();
    const pile = stacked.length > 1 ? ` · ${stacked.length} on this spot` : "";
    scatterTip.hidden = false;
    scatterTip.textContent = `${name} · ${kindLabel(kindFor(frame))} · ${percent(frame.confidence)} · ${formatDistance(Number(frame[scoreKey]))}${pile}`;
    const wrap = canvas.parentElement.getBoundingClientRect();
    let left = event.clientX - wrap.left + 14;
    let top = event.clientY - wrap.top + 14;
    if (left + scatterTip.offsetWidth > wrap.width - 4) {
      left = event.clientX - wrap.left - scatterTip.offsetWidth - 8;
    }
    if (top + scatterTip.offsetHeight > wrap.height - 4) {
      top = event.clientY - wrap.top - scatterTip.offsetHeight - 8;
    }
    scatterTip.style.left = `${Math.max(4, left)}px`;
    scatterTip.style.top = `${Math.max(4, top)}px`;
  }

  function hideTip() {
    hoverKey = "";
    if (scatterTip) scatterTip.hidden = true;
  }

  function zoomAt(px, py, factor) {
    const view = layout();
    const x = view.xData(px);
    const y = view.yData(py);
    const bounds = view.bounds;
    clampBounds({
      x0: x - (x - bounds.x0) * factor,
      x1: x + (bounds.x1 - x) * factor,
      y0: y - (y - bounds.y0) * factor,
      y1: y + (bounds.y1 - y) * factor,
    });
    draw(visibleFrames());
  }

  function panPixels(dx, dy) {
    const view = layout();
    const bounds = view.bounds;
    const xSpan = bounds.x1 - bounds.x0;
    const ySpan = bounds.y1 - bounds.y0;
    clampBounds({
      x0: bounds.x0 - (dx / view.plotW) * xSpan,
      x1: bounds.x1 - (dx / view.plotW) * xSpan,
      y0: bounds.y0 + (dy / view.plotH) * ySpan,
      y1: bounds.y1 + (dy / view.plotH) * ySpan,
    });
    draw(visibleFrames());
  }

  function applyRubber(state, local) {
    const view = layout();
    const x0 = view.xData(Math.min(state.x, local.x));
    const x1 = view.xData(Math.max(state.x, local.x));
    const yA = view.yData(Math.min(state.y, local.y));
    const yB = view.yData(Math.max(state.y, local.y));
    clampBounds({
      x0,
      x1,
      y0: Math.min(yA, yB),
      y1: Math.max(yA, yB),
    });
    draw(visibleFrames());
  }

  function draw(rows) {
    const view = layout();
    const width = view.width;
    const height = view.height;
    if (!width || !height) return;
    const { pad, plotW, plotH, bounds, xOf, yOf } = view;
    const dpr = window.devicePixelRatio || 1;
    const nextW = Math.round(width * dpr);
    const nextH = Math.round(height * dpr);
    if (canvas.width !== nextW || canvas.height !== nextH) {
      canvas.width = nextW;
      canvas.height = nextH;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#243044";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);
    ctx.fillStyle = "#8b98ab";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    const xDigits = bounds.x1 - bounds.x0 < 0.08 ? 3 : 2;
    axisTicks(bounds.x0, bounds.x1, 4).forEach((tick) => {
      const x = xOf(tick);
      if (x < pad.left - 1 || x > pad.left + plotW + 1) return;
      ctx.fillText(tick.toFixed(xDigits), x, height - 10);
    });
    ctx.textAlign = "right";
    axisTicks(bounds.y0, bounds.y1, 3).forEach((tick) => {
      const y = yOf(tick);
      if (y < pad.top - 1 || y > pad.top + plotH + 1) return;
      ctx.fillText(formatDistance(tick), pad.left - 6, y + 4);
    });
    if (zoomNote) {
      const times = 0.5 / (bounds.x1 - bounds.x0);
      const full = zoom.y1 == null && times < 1.05;
      zoomNote.textContent = full
        ? "Scroll to zoom toward the cursor. Drag to pan. Shift-drag a box around a cluster. Next walks photos that still sit on the same spot."
        : `Zoomed ${times.toFixed(1)}×. Scroll to zoom. Drag to pan. Shift-drag a box. Next walks photos on the same spot.`;
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#e7edf5";
    ctx.beginPath();
    ctx.moveTo(xOf(confidenceCut()), pad.top);
    ctx.lineTo(xOf(confidenceCut()), pad.top + plotH);
    ctx.moveTo(pad.left, yOf(distanceCut()));
    ctx.lineTo(pad.left + plotW, yOf(distanceCut()));
    ctx.stroke();
    ctx.setLineDash([]);
    if (keepInput.checked) {
      ctx.strokeStyle = "#3ee0c5";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xOf(keepCut()), pad.top);
      ctx.lineTo(xOf(keepCut()), pad.top + plotH);
      ctx.stroke();
    }
    points = [];
    const rank = { kept: 0, uncertain: 1, others: 2, misidentified: 3 };
    const ordered = rows.concat(probes).slice().sort(
      (a, b) => (rank[kindFor(a)] || 0) - (rank[kindFor(b)] || 0)
    );
    const zoomTimes = 0.5 / (bounds.x1 - bounds.x0);
    const scale = Math.min(2.6, Math.max(1, Math.sqrt(zoomTimes)));
    const ringColor = {
      others: "#f0b429",
      uncertain: "#8b98ab",
      misidentified: "#ff5d73",
    };
    ordered.forEach((frame) => {
      const x = xOf(Number(frame.confidence));
      const y = yOf(Number(frame[scoreKey]));
      const radius = (frame.split === "probe" ? 4.4 : 3.2) * scale;
      if (x < pad.left - radius || x > pad.left + plotW + radius) return;
      if (y < pad.top - radius || y > pad.top + plotH + radius) return;
      const kind = kindFor(frame);
      const point = { x, y, frame, list: frame.split === "probe" ? probes : rows };
      points.push(point);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (frame.true === "metal") ctx.fillStyle = "#f0b429";
      else if (frame.true === "plastic") ctx.fillStyle = "#3ee0c5";
      else ctx.fillStyle = "#c084fc";
      ctx.fill();
      ctx.lineWidth = 1.15;
      ctx.strokeStyle = "#07090d";
      ctx.stroke();
      if (ringColor[kind]) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 1.7, 0, Math.PI * 2);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = ringColor[kind];
        ctx.stroke();
      }
      if (hoverKey && hoverKey === `${frame.split}\0${frame.path}`) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 3.4, 0, Math.PI * 2);
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = "#e7edf5";
        ctx.stroke();
      }
    });
    ctx.restore();
    if (rubber) {
      const x = Math.min(rubber.x0, rubber.x1);
      const y = Math.min(rubber.y0, rubber.y1);
      ctx.save();
      ctx.fillStyle = "rgba(62, 224, 197, 0.14)";
      ctx.strokeStyle = "#3ee0c5";
      ctx.lineWidth = 1;
      ctx.fillRect(x, y, Math.abs(rubber.x1 - rubber.x0), Math.abs(rubber.y1 - rubber.y0));
      ctx.strokeRect(x, y, Math.abs(rubber.x1 - rubber.x0), Math.abs(rubber.y1 - rubber.y0));
      ctx.restore();
    }
  }

  function render() {
    if (!payload) return;
    const max = axisMax(scoreKey);
    distanceInput.max = String(max);
    distanceNumber.max = String(max);
    if (Number(distanceInput.value) > max) {
      distanceInput.value = String(max);
      distanceNumber.value = trimNumber(max);
    }
    const rows = visibleFrames();
    const ruleNow = currentRule();
    const measured = measure(rows, ruleNow, true);
    const counts = measured.stats;
    measured.newly.sort((a, b) => Number(b[scoreKey]) - Number(a[scoreKey]));
    measured.keptRows.sort((a, b) => Number(b.confidence) - Number(a.confidence));
    setFigure("accepted", share(counts.accepted, rows.length), `${countLabel(counts.accepted)} of ${countLabel(rows.length)}`);
    setFigure("uncertain", share(counts.uncertain, rows.length), `${countLabel(counts.uncertain)} of ${countLabel(rows.length)}`);
    setFigure("others", share(counts.others, rows.length), `${countLabel(counts.knownOthers)} of ${countLabel(counts.known)} known`);
    setFigure("kept", share(counts.kept, rows.length), ruleNow.keepOn ? `${countLabel(counts.kept)} of ${countLabel(rows.length)}` : "keep is off");
    setFigure("held", share(counts.uncertain + counts.others, rows.length), `${countLabel(counts.uncertain + counts.others)} of ${countLabel(rows.length)}`);
    setFigure(
      "caught",
      share(counts.caught, counts.wrong),
      counts.wrong ? `${countLabel(counts.caught)} of ${countLabel(counts.wrong)} misses` : "no misses"
    );
    setFigure("still", share(counts.missed, counts.known), `${countLabel(counts.missed)} of ${countLabel(counts.known)} known`);
    setFigure("held-right", share(counts.heldRight, counts.known), `${countLabel(counts.heldRight)} of ${countLabel(counts.known)} known`);
    setFigure(
      "accuracy",
      share(counts.acceptedCorrect, counts.accepted),
      counts.accepted ? `${countLabel(counts.acceptedCorrect)} of ${countLabel(counts.accepted)} accepted` : "none accepted"
    );
    setFigure(
      "balanced",
      counts.balanced === null ? "—" : (counts.balanced * 100).toFixed(2) + "%",
      "on the accepted"
    );
    setFigure(
      "uncertain-right",
      share(counts.uncertainRight, rows.length),
      counts.uncertain ? `${countLabel(counts.uncertainRight)} of ${countLabel(counts.uncertain)} uncertain` : "none uncertain"
    );
    setFigure(
      "uncertain-wrong",
      share(counts.uncertainWrong, rows.length),
      counts.uncertain ? `${countLabel(counts.uncertainWrong)} of ${countLabel(counts.uncertain)} uncertain` : "none uncertain"
    );
    fillOpenMatrix(measured.matrix);
    note.textContent = SPLIT_NOTES[split] || "";
    const label = SCORE_LABELS[scoreKey] || "Distance";
    openSet.querySelector("[data-distance-label]").textContent = label;
    disagreeNote.textContent = `${countLabel(counts.clash)} photos in this split sit closer to the other class.`;
    const mark = payload.suggestions ? payload.suggestions[scoreKey] : null;
    const markText = Number.isFinite(Number(mark)) ? formatDistance(Number(mark)) : "none";
    openSet.querySelector("[data-use-mark]").textContent = `Validation 99% (${markText})`;
    const keepText = ruleNow.keepOn
      ? `A photo at or above ${formatKeep(ruleNow.keep)} confidence keeps the winning class, even past the distance mark. `
      : "The keep line is off, so distance can reject a confident photo. ";
    rule.textContent =
      `Others when ${label.toLowerCase()} is above ${formatDistance(ruleNow.distance)}. ` +
      `Uncertain when confidence is under ${ruleNow.floor.toFixed(2)} and the photo is still inside a cloud. ` +
      keepText +
      (ruleNow.clash
        ? "A photo whose nearest cloud is the other class is others too, unless the keep line saves it."
        : "Otherwise the winning class.");
    paintStrip(measured.newly);
    if (ruleNow.keepOn) paintList(keptStrip, keptCount, measured.keptRows);
    else {
      keptStrip.replaceChildren();
      keptCount.textContent = "Keep is off";
    }
    paintProbes();
    draw(rows);
  }

  function markManual() {
    fitNote.textContent = "Manual. The sliders are what you set. Fit on validation can replace the keep line, and the distance mark if you ask it to.";
  }

  function setKeep(value, fromFit) {
    const next = Math.max(confidenceCut(), snapKeep(value));
    keepValue.value = String(next);
    keepNumber.value = formatKeep(next);
    if (!fromFit) markManual();
    render();
  }

  function setConfidence(value, fromFit) {
    const next = snapConfidence(value);
    confidenceInput.value = String(next);
    confidenceNumber.value = next.toFixed(2);
    const raised = Math.max(next, snapKeep(keepValue.value));
    keepValue.value = String(raised);
    keepNumber.value = formatKeep(raised);
    if (!fromFit) markManual();
    render();
  }

  function setDistance(value, fromFit) {
    const max = axisMax(scoreKey);
    let next = Number(value);
    if (!Number.isFinite(next)) next = 0;
    next = Math.min(max, Math.max(0, next));
    distanceInput.max = String(max);
    distanceNumber.max = String(max);
    distanceInput.value = String(next);
    distanceNumber.value = trimNumber(next);
    if (!fromFit) markManual();
    render();
  }

  function goalKey(stats, goal) {
    const balanced = stats.balanced === null ? -1 : stats.balanced;
    const accuracy = stats.accepted ? stats.acceptedCorrect / stats.accepted : -1;
    if (goal === "balanced") return [balanced, -stats.missed, -stats.heldRight];
    if (goal === "accuracy") return [accuracy, stats.accepted, -stats.missed];
    if (goal === "still") return [-stats.missed, balanced, -stats.heldRight];
    return [stats.caught - stats.heldRight, balanced, -stats.hold];
  }

  function betterCandidate(candidate, best) {
    if (!best) return true;
    const candidateOk = candidate.hold <= HOLD_CAP;
    const bestOk = best.hold <= HOLD_CAP;
    if (candidateOk !== bestOk) return candidateOk;
    if (!candidateOk && candidate.hold !== best.hold) return candidate.hold < best.hold;
    for (let index = 0; index < candidate.key.length; index += 1) {
      if (candidate.key[index] !== best.key[index]) return candidate.key[index] > best.key[index];
    }
    return false;
  }

  function quantile(sorted, portion) {
    if (!sorted.length) return null;
    const index = (sorted.length - 1) * portion;
    const low = Math.floor(index);
    const high = Math.ceil(index);
    if (low === high) return sorted[low];
    return sorted[low] * (high - index) + sorted[high] * (index - low);
  }

  function fitKeep() {
    const rows = frames.filter((frame) => frame.split === "validation");
    if (!rows.length) {
      fitNote.textContent = "No validation photos to fit on.";
      return;
    }
    const floor = confidenceCut();
    const distances = [distanceCut()];
    if (fitWhat.value === "both") {
      const values = rows
        .filter((frame) => known(frame) && frame.pred === frame.true)
        .map((frame) => Number(frame[scoreKey]))
        .filter(Number.isFinite)
        .sort((left, right) => left - right);
      [0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995].forEach((portion) => {
        const value = quantile(values, portion);
        if (value !== null) distances.push(value);
      });
      const suggestion = payload.suggestions ? Number(payload.suggestions[scoreKey]) : NaN;
      if (Number.isFinite(suggestion)) distances.push(suggestion);
      distances.push(axisMax(scoreKey));
    }
    const uniqueDistance = [];
    distances.forEach((value) => {
      const rounded = Math.round(value * 1e6) / 1e6;
      if (!uniqueDistance.some((item) => Math.abs(item - rounded) < 1e-6)) uniqueDistance.push(rounded);
    });
    const keeps = [{ on: false, value: floor }];
    for (let step = Math.round(floor * 100); step <= 99; step += 1) {
      keeps.push({ on: true, value: step / 100 });
    }
    const goal = fitGoal.value;
    let best = null;
    uniqueDistance.forEach((distance) => {
      keeps.forEach((keep) => {
        const stats = measure(rows, {
          keepOn: keep.on,
          keep: keep.value,
          distance,
          key: scoreKey,
          floor,
          clash: disagreeInput.checked,
        }, false).stats;
        const candidate = {
          keepOn: keep.on,
          keep: keep.value,
          distance,
          hold: stats.hold,
          stats,
          key: goalKey(stats, goal),
        };
        if (betterCandidate(candidate, best)) best = candidate;
      });
    });
    keepInput.checked = best.keepOn;
    keepValue.value = String(best.keep);
    keepNumber.value = formatKeep(best.keep);
    if (fitWhat.value === "both") {
      const max = axisMax(scoreKey);
      distanceInput.max = String(max);
      distanceNumber.max = String(max);
      distanceInput.value = String(best.distance);
      distanceNumber.value = trimNumber(best.distance);
    }
    render();
    const names = {
      balanced: "balanced accuracy",
      accuracy: "accuracy on accepted",
      still: "fewest still wrong",
      tradeoff: "catching misses without holding correct photos",
    };
    const keepLine = best.keepOn ? `Keep at ${formatKeep(best.keep)}.` : "Keep is off.";
    const distanceLine = fitWhat.value === "both"
      ? ` Distance mark ${formatDistance(best.distance)}.`
      : " Distance mark left where it was.";
    const balancedText = best.stats.balanced === null
      ? "—"
      : `${(best.stats.balanced * 100).toFixed(2)}%`;
    fitNote.textContent =
      `Fit on validation for ${names[goal] || goal}. ${keepLine}${distanceLine} ` +
      `Held ${share(best.stats.uncertain + best.stats.others, best.stats.n)} ` +
      `(${countLabel(best.stats.uncertain + best.stats.others)} of ${countLabel(best.stats.n)}). ` +
      `Balanced ${balancedText}. Accuracy ${share(best.stats.acceptedCorrect, best.stats.accepted)}. ` +
      `Still wrong ${countLabel(best.stats.missed)}. ` +
      (best.hold <= HOLD_CAP
        ? "Held at most 5% of validation. "
        : "No candidate held 5% or less, so this is the smallest hold. ") +
      "Test did not pick this.";
  }

  function useMark() {
    const mark = payload && payload.suggestions ? Number(payload.suggestions[scoreKey]) : NaN;
    if (!Number.isFinite(mark)) {
      setDistance(Number(distanceInput.value) || 0);
      return;
    }
    setDistance(mark);
  }

  function showView() {
    const frame = viewList[viewIndex];
    if (!frame) return;
    const call = judge(frame, currentRule());
    dialogTitle.textContent = frame.path || "Photo";
    dialogImage.src = fileUrl(frame.path);
    dialogImage.alt = frame.path || "photo";
    const trueName = frame.true || "unlabeled";
    dialogCaption.textContent =
      `${call.decision}${call.rescued ? " · kept past distance" : ""} · true ${trueName} · pred ${frame.resin || frame.pred} ` +
      `${percent(frame.confidence)} · cloud ${formatDistance(frame.cloud)} · ` +
      `call ${formatDistance(frame.call)} · k-NN ${formatDistance(frame.knn)} · ` +
      `Mahalanobis ${formatDistance(frame.maha)}` +
      (frame.disagree ? ` · nearest ${frame.nearest}` : "");
  }

  function openFrame(frame, list) {
    viewList = list;
    viewIndex = Math.max(0, list.indexOf(frame));
    showView();
    if (!dialog.open) dialog.showModal();
  }

  openSet.querySelectorAll("[data-split-value]").forEach((button) => {
    button.addEventListener("click", () => {
      split = button.dataset.splitValue;
      openSet.querySelectorAll("[data-split-value]").forEach((item) => {
        item.setAttribute("aria-pressed", item === button ? "true" : "false");
      });
      render();
    });
  });
  confidenceInput.addEventListener("input", () => setConfidence(confidenceInput.value));
  confidenceNumber.addEventListener("change", () => setConfidence(confidenceNumber.value));
  distanceInput.addEventListener("input", () => setDistance(distanceInput.value));
  distanceNumber.addEventListener("change", () => setDistance(distanceNumber.value));
  scoreSelect.addEventListener("change", () => {
    scoreKey = scoreSelect.value;
    resetZoom();
    const max = axisMax(scoreKey);
    distanceInput.max = String(max);
    distanceNumber.max = String(max);
    useMark();
  });
  disagreeInput.addEventListener("change", () => {
    markManual();
    render();
  });
  keepValue.addEventListener("input", () => setKeep(keepValue.value));
  keepNumber.addEventListener("change", () => setKeep(keepNumber.value));
  keepInput.addEventListener("change", () => {
    markManual();
    render();
  });
  openSet.querySelector("[data-fit]").addEventListener("click", fitKeep);
  openSet.querySelector("[data-use-frozen]").addEventListener("click", () => {
    setConfidence(openSet.dataset.frozen || "0.85");
  });
  openSet.querySelector("[data-use-mark]").addEventListener("click", useMark);
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const local = eventLocal(event);
    pointer = {
      id: event.pointerId,
      x: local.x,
      y: local.y,
      lastX: local.x,
      lastY: local.y,
      moved: false,
      box: event.shiftKey,
    };
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  canvas.addEventListener("pointermove", (event) => {
    const local = eventLocal(event);
    if (pointer && pointer.id === event.pointerId) {
      const travel = Math.hypot(local.x - pointer.x, local.y - pointer.y);
      if (travel > 4) pointer.moved = true;
      if (pointer.box) {
        rubber = { x0: pointer.x, y0: pointer.y, x1: local.x, y1: local.y };
        draw(visibleFrames());
      } else if (pointer.moved) {
        canvas.classList.add("panning");
        const dx = local.x - pointer.lastX;
        const dy = local.y - pointer.lastY;
        pointer.lastX = local.x;
        pointer.lastY = local.y;
        panPixels(dx, dy);
      }
      return;
    }
    const hit = nearestPoint(local.x, local.y, 12);
    placeTip(hit.best, hit.stacked, event);
    const key = hit.best ? `${hit.best.frame.split}\0${hit.best.frame.path}` : "";
    if (key !== hoverKey) {
      hoverKey = key;
      draw(visibleFrames());
      if (hit.best) placeTip(hit.best, hit.stacked, event);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!pointer || pointer.id !== event.pointerId) return;
    const local = eventLocal(event);
    const state = pointer;
    pointer = null;
    rubber = null;
    canvas.classList.remove("panning");
    if (state.box) {
      if (state.moved) applyRubber(state, local);
      else draw(visibleFrames());
      return;
    }
    if (!state.moved) {
      const hit = nearestPoint(state.x, state.y, 14);
      if (hit.best) {
        const pile = hit.stacked.map((point) => point.frame);
        openFrame(hit.best.frame, pile.length > 1 ? pile : hit.best.list);
      }
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (pointer) return;
    if (!hoverKey && (!scatterTip || scatterTip.hidden)) return;
    hideTip();
    draw(visibleFrames());
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const local = eventLocal(event);
    const dy = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    const factor = Math.min(2.4, Math.max(1 / 2.4, Math.exp(dy * 0.012)));
    zoomAt(local.x, local.y, factor);
  }, { passive: false });
  openSet.querySelector("[data-zoom-reset]").addEventListener("click", () => {
    resetZoom();
    hideTip();
    draw(visibleFrames());
  });
  openSet.querySelectorAll("[data-list-source]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.listSource;
      listSources[key] = !listSources[key];
      button.setAttribute("aria-pressed", listSources[key] ? "true" : "false");
      listCap = LIST_STEP;
      render();
    });
  });
  openSet.querySelectorAll("[data-list-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      listFilter = button.dataset.listFilter;
      openSet.querySelectorAll("[data-list-filter]").forEach((item) => {
        item.setAttribute("aria-pressed", item === button ? "true" : "false");
      });
      listCap = LIST_STEP;
      render();
    });
  });
  openSet.querySelector("[data-list-sort]").addEventListener("change", (event) => {
    listSort = event.target.value;
    listCap = LIST_STEP;
    render();
  });
  if (listMore) {
    listMore.addEventListener("click", () => {
      listCap += LIST_STEP;
      render();
    });
  }
  dialog.querySelector("[data-open-close]").addEventListener("click", () => dialog.close());
  dialog.querySelector("[data-open-prev]").addEventListener("click", () => {
    if (!viewList.length) return;
    viewIndex = (viewIndex - 1 + viewList.length) % viewList.length;
    showView();
  });
  dialog.querySelector("[data-open-next]").addEventListener("click", () => {
    if (!viewList.length) return;
    viewIndex = (viewIndex + 1) % viewList.length;
    showView();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      dialog.querySelector("[data-open-prev]").click();
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      dialog.querySelector("[data-open-next]").click();
    }
  });
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (payload) draw(visibleFrames());
    }).observe(canvas.parentElement);
  }
  probeForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const button = probeForm.querySelector("button");
    const body = new FormData(probeForm);
    body.set("model", openSet.dataset.model || "");
    button.disabled = true;
    probeStatus.textContent = "Scoring…";
    fetch(openSet.dataset.probeUrl, { method: "POST", body })
      .then((response) => response.json().then((data) => ({ ok: response.ok, data })))
      .then((result) => {
        if (!result.ok || !result.data.ok) {
          probeStatus.textContent = (result.data && result.data.error) || "Could not score those photos.";
          return;
        }
        const rows = result.data.frames || [];
        const folder = (body.get("folder") || "").trim();
        rows.forEach((frame) => {
          frame.split = "probe";
          const index = probes.findIndex((item) => item.path === frame.path);
          if (index >= 0) probes[index] = frame;
          else probes.push(frame);
        });
        const where = folder ? ` from ${folder}` : "";
        const skipped = result.data.skipped || [];
        let skipNote = "";
        if (skipped.length === 1) {
          skipNote = ` Skipped 1 empty or unreadable file: ${skipped[0]}.`;
        } else if (skipped.length > 1) {
          skipNote = ` Skipped ${skipped.length} empty or unreadable files.`;
        }
        probeStatus.textContent = `Scored ${rows.length}${where}.${skipNote} Cards stay until this page reloads.`;
        probeForm.reset();
        render();
      })
      .catch(() => {
        probeStatus.textContent = "Could not score those photos.";
      })
      .finally(() => {
        button.disabled = false;
      });
  });

  fetch(openSet.dataset.scoresUrl)
    .then((response) => {
      if (!response.ok) throw new Error("scores");
      return response.json();
    })
    .then((data) => {
      payload = data;
      frames = data.frames || [];
      if (status) status.hidden = true;
      const frozen = snapConfidence(openSet.dataset.frozen || "0.85");
      confidenceInput.value = String(frozen);
      confidenceNumber.value = frozen.toFixed(2);
      scoreKey = "cloud";
      useMark();
    })
    .catch(() => {
      if (status) status.textContent = "Could not load scores.";
    });
}
