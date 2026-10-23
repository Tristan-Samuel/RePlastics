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
  document.querySelectorAll('form[action$="/jobs/start"] button').forEach((button) => {
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

  function card(frame, showSplit) {
    const article = document.createElement("article");
    article.className = "frame";
    if (frame.image) {
      const link = document.createElement("a");
      link.href = fileUrl(frame.image);
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = frame.path.split("/").pop();
      img.src = link.href;
      link.appendChild(img);
      article.appendChild(link);
    }
    const line = document.createElement("p");
    const bits = [
      frame.true + " → " + frame.pred,
      percent(frame.score),
    ];
    if (frame.resin && frame.resin !== "unknown") bits.push(frame.resin);
    if (showSplit) bits.push(frame.split);
    line.textContent = bits.join(" · ");
    const path = document.createElement("p");
    path.className = "path";
    path.textContent = frame.path.split("/").pop();
    article.append(line, path);
    return article;
  }

  function fillCards(host, rows, showSplit) {
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
    rows.forEach((frame) => fragment.appendChild(card(frame, showSplit)));
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
    fillCards(bandHost, shown, showSplit);
  }

  function fillBucket(name, rows, showSplit) {
    const host = decide.querySelector(`[data-bucket="${name}"]`);
    host.querySelector("[data-bucket-count]").textContent = rows.length.toLocaleString("en-US");
    fillCards(host, rows, showSplit);
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
  }

  function openViewer() {
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
