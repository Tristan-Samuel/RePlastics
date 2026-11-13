const root = document.querySelector("[data-pipeline]");
if (root) {
  const checks = JSON.parse(document.querySelector("#pipeline-checks").textContent);
  const modelChoices = JSON.parse(document.querySelector("#pipeline-models").textContent);
  const scoredAtLoad = new Set(JSON.parse(document.querySelector("#pipeline-scored").textContent));
  const board = root.querySelector("[data-board]");
  const canvas = root.querySelector("[data-canvas]");
  const svg = root.querySelector("[data-wires]");
  const inspector = root.querySelector("[data-inspector]");
  const flowNote = root.querySelector("[data-flow-note]");
  const figures = document.querySelector("[data-figures]");
  const stepsHost = document.querySelector("[data-steps]");
  const moreButton = document.querySelector("[data-list-more]");
  const listNote = document.querySelector("[data-list-note]");
  const splitNote = document.querySelector("[data-split-note]");
  const recipe = document.querySelector("[data-recipe]");
  const dialog = document.querySelector("[data-viewer]");
  const scoreForm = document.querySelector("#pipeline-score");
  const scoreButton = document.querySelector("[data-score-button]");
  const scoreNote = document.querySelector("[data-score-note]");
  const MODES = [
    ["cloud", "cloud distance"],
    ["call", "call distance"],
    ["knn", "k-NN distance"],
    ["maha", "Mahalanobis"],
  ];
  const FLOORS = [0.5, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99];
  const KEEPS = [0.9, 0.93, 0.95, 0.97, 0.99, 0.995];
  const PORTIONS = [0.9, 0.95, 0.97, 0.99, 0.995];
  const HOLD_CAP = 0.05;
  const LIST_STEP = 96;
  const ADD_TYPES = { belt: true, model: true, merge: true, sink: true, input: true };
  const draftKey = "pipeline-draft:" + root.dataset.dataset;
  const graphNameKey = "pipeline-graph:" + root.dataset.dataset;
  const resinLib = window.labResin || {
    order: ["HDPE", "PET", "PP", "OTHERS"],
    colors: { HDPE: "#3ee0c5", PET: "#6aa7ff", PP: "#f0b429", OTHERS: "#c084fc" },
    fromPath: function () { return ""; },
  };

  let catalog = checks.slice();
  const loaded = new Map();
  let graph = loadDraft() || starter();
  let serial = 20;
  let selectedId = graph.nodes[0] ? graph.nodes[0].id : "";
  let pending = null;
  let split = "validation";
  let portFilter = null;
  let viewNodes = [];
  let outcomeFilter = "all";
  let classFilter = "any";
  let minConfidence = 0;
  let equalClasses = false;
  let hideMetal = false;
  let showDistance = false;
  let listCap = LIST_STEP;
  let focusAfterScore = [];
  let inspectorStale = false;
  let belt = null;
  let frames = [];
  let outcome = null;
  let drag = null;
  let palette = "";

  bumpSerial();
  bindChrome();
  paintBoard();
  paintInspector();
  loadBelt();

  function checkInfo(id) {
    return catalog.find((item) => item.id === id) || catalog[0];
  }

  function modelChoice(rel) {
    return modelChoices.find((item) => item.rel === rel) || null;
  }

  function modelTitle(rel) {
    if (!rel) return "Pick a checkpoint";
    const found = modelChoice(rel);
    return found ? found.title : rel;
  }

  function modelClasses(rel) {
    const payload = loaded.get(rel);
    if (payload && Array.isArray(payload.classes) && payload.classes.length) return payload.classes.slice();
    const found = modelChoice(rel);
    if (found && Array.isArray(found.classes) && found.classes.length) return found.classes.slice();
    return ["metal", "plastic"];
  }

  function nodeById(id) {
    return graph.nodes.find((node) => node.id === id) || null;
  }

  function titleOf(node) {
    if (!node) return "Missing card";
    if (node.type === "input") return "Input";
    if (node.type === "belt") return checkInfo(node.check).label;
    if (node.type === "model") return node.title || modelTitle(node.model);
    if (node.type === "merge") return node.title || "Merge";
    return node.name || "Sink";
  }

  function portsOf(node) {
    if (!node) return [];
    if (node.type === "input" || node.type === "merge") return ["out"];
    if (node.type === "belt") return ["occupied", "empty"];
    if (node.type === "model") return modelClasses(node.model).concat(["uncertain", "others"]);
    return [];
  }

  function starter() {
    const blob = checkInfo("largest_blob");
    const chroma = checkInfo("chroma");
    const model = root.dataset.defaultModel || "";
    const classes = modelClasses(model);
    if (classes.indexOf("HDPE") !== -1 && classes.indexOf("PET") !== -1 && classes.indexOf("PP") !== -1) {
      return resinStarter(model, classes, blob, chroma);
    }
    return {
      nodes: [
        { id: "in", type: "input", x: 24, y: 200 },
        { id: "blob", type: "belt", check: "largest_blob", cut: blob.cut, emptyWhen: blob.empty_when, x: 280, y: 80 },
        { id: "chroma", type: "belt", check: "chroma", cut: chroma.cut, emptyWhen: chroma.empty_when, x: 280, y: 320 },
        { id: "join", type: "merge", title: "Occupied", x: 560, y: 180 },
        { id: "resin", type: "model", model: model, confidence: 0.85, mode: "cloud", distance: 0.5, keepOn: false, keep: 0.95, clash: false, x: 840, y: 120 },
        { id: "cans", type: "model", model: "", title: "Cans", confidence: 0.85, mode: "knn", distance: 0.5, keepOn: false, keep: 0.97, clash: false, x: 1120, y: 300 },
        { id: "metal", type: "sink", name: "Sort metal", role: "metal", x: 1420, y: 24 },
        { id: "plastic", type: "sink", name: "Sort plastic", role: "plastic", x: 1420, y: 170 },
        { id: "empty", type: "sink", name: "Empty", role: "empty", x: 560, y: 420 },
        { id: "hold", type: "sink", name: "Hold", role: "hold", x: 1420, y: 360 },
      ],
      wires: [
        { id: "w1", from: "in", port: "out", to: "blob" },
        { id: "w2", from: "blob", port: "occupied", to: "join" },
        { id: "w3", from: "blob", port: "empty", to: "chroma" },
        { id: "w4", from: "chroma", port: "occupied", to: "join" },
        { id: "w5", from: "chroma", port: "empty", to: "empty" },
        { id: "w6", from: "join", port: "out", to: "resin" },
        { id: "w7", from: "resin", port: "metal", to: "metal" },
        { id: "w8", from: "resin", port: "plastic", to: "plastic" },
        { id: "w9", from: "resin", port: "uncertain", to: "cans" },
        { id: "w10", from: "resin", port: "others", to: "cans" },
        { id: "w11", from: "cans", port: "metal", to: "metal" },
        { id: "w12", from: "cans", port: "plastic", to: "plastic" },
        { id: "w13", from: "cans", port: "uncertain", to: "hold" },
        { id: "w14", from: "cans", port: "others", to: "hold" },
      ],
    };
  }

  function resinStarter(model, classes, blob, chroma) {
    const sinks = classes.map((name, index) => ({
      id: name.toLowerCase(),
      type: "sink",
      name: name,
      role: name,
      x: 1180,
      y: 24 + index * 150,
    }));
    const wires = [
      { id: "w1", from: "in", port: "out", to: "blob" },
      { id: "w2", from: "blob", port: "occupied", to: "join" },
      { id: "w3", from: "blob", port: "empty", to: "chroma" },
      { id: "w4", from: "chroma", port: "occupied", to: "join" },
      { id: "w5", from: "chroma", port: "empty", to: "empty" },
      { id: "w6", from: "join", port: "out", to: "resin" },
      { id: "w7", from: "resin", port: "uncertain", to: "hold" },
      { id: "w8", from: "resin", port: "others", to: "hold" },
    ];
    classes.forEach((name) => {
      wires.push({
        id: "w-" + name.toLowerCase(),
        from: "resin",
        port: name,
        to: name.toLowerCase(),
      });
    });
    return {
      nodes: [
        { id: "in", type: "input", x: 24, y: 200 },
        { id: "blob", type: "belt", check: "largest_blob", cut: blob.cut, emptyWhen: blob.empty_when, x: 280, y: 80 },
        { id: "chroma", type: "belt", check: "chroma", cut: chroma.cut, emptyWhen: chroma.empty_when, x: 280, y: 360 },
        { id: "join", type: "merge", title: "Occupied", x: 560, y: 180 },
        { id: "resin", type: "model", model: model, confidence: 0.85, mode: "cloud", distance: 0.5, keepOn: false, keep: 0.95, clash: false, x: 840, y: 140 },
        { id: "empty", type: "sink", name: "Empty", role: "empty", x: 560, y: 460 },
        { id: "hold", type: "sink", name: "Hold", role: "hold", x: 1180, y: 24 + classes.length * 150 },
      ].concat(sinks),
      wires: wires,
    };
  }

  function loadDraft() {
    try {
      const raw = JSON.parse(localStorage.getItem(draftKey) || "null");
      if (!raw || !Array.isArray(raw.nodes) || !Array.isArray(raw.wires)) return null;
      return raw;
    } catch (error) {
      return null;
    }
  }

  function saveDraft() {
    localStorage.setItem(draftKey, JSON.stringify(graph));
  }

  function bumpSerial() {
    graph.nodes.concat(graph.wires).forEach((item) => {
      const match = /(\d+)$/.exec(item.id || "");
      if (match) serial = Math.max(serial, Number(match[1]));
    });
  }

  function nextId(prefix) {
    serial += 1;
    return prefix + serial;
  }

  function bindChrome() {
    root.querySelectorAll("[data-add]").forEach((button) => {
      button.draggable = true;
      button.addEventListener("dragstart", (event) => {
        palette = button.dataset.add || "";
        event.dataTransfer.setData("text/plain", palette);
        event.dataTransfer.effectAllowed = "copy";
      });
      button.addEventListener("dragend", () => {
        palette = "";
        board.classList.remove("drop-ready");
      });
      button.addEventListener("click", () => addNode(button.dataset.add));
    });
    board.addEventListener("dragover", (event) => {
      if (!ADD_TYPES[palette]) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      board.classList.add("drop-ready");
    });
    board.addEventListener("dragleave", (event) => {
      if (!board.contains(event.relatedTarget)) board.classList.remove("drop-ready");
    });
    board.addEventListener("drop", (event) => {
      const type = ADD_TYPES[palette] ? palette : "";
      palette = "";
      board.classList.remove("drop-ready");
      if (!type) return;
      event.preventDefault();
      const origin = canvas.getBoundingClientRect();
      addNode(type, event.clientX - origin.left - 110, event.clientY - origin.top - 20);
    });
    document.addEventListener("lab-job-done", (event) => {
      const title = (event.detail && event.detail.title) || "";
      if (title.indexOf("Pipeline") === -1) return;
      event.preventDefault();
      const state = event.detail || {};
      refreshScores().then(() => {
        if (focusAfterScore.length) showFromNodes(focusAfterScore);
        focusAfterScore = [];
        if (!scoreNote) return;
        scoreNote.textContent = state.exit_code
          ? (state.status_line || "Scoring stopped.")
          : "Scores are in. Move a card or a cut without scoring again.";
      });
    });
    root.querySelector("[data-tidy]").addEventListener("click", () => {
      tidy();
      saveDraft();
      paintBoard();
      run();
    });
    root.querySelector("[data-save-form]").addEventListener("submit", saveGraph);
    const loadGraphButton = root.querySelector("[data-graph-load]");
    if (loadGraphButton) loadGraphButton.addEventListener("click", () => {
      const name = root.querySelector("[data-graph-pick]").value;
      if (!name) {
        applyGraph(loadDraft(), "");
        flowNote.textContent = "Loaded the draft saved in this browser.";
        return;
      }
      loadGraph(name);
    });
    const savedName = localStorage.getItem(graphNameKey) || "";
    const graphPick = root.querySelector("[data-graph-pick]");
    if (savedName && Array.from(graphPick.options).some((option) => option.value === savedName)) {
      graphPick.value = savedName;
    }
    document.querySelectorAll("[data-split-value]").forEach((button) => {
      button.addEventListener("click", () => {
        split = button.dataset.splitValue;
        portFilter = null;
        listCap = LIST_STEP;
        document.querySelectorAll("[data-split-value]").forEach((item) => {
          item.setAttribute("aria-pressed", item === button ? "true" : "false");
        });
        run();
      });
    });
    if (moreButton) moreButton.addEventListener("click", () => {
      listCap += LIST_STEP;
      paintList();
    });
    document.querySelectorAll("[data-outcome]").forEach((button) => {
      button.addEventListener("click", () => {
        outcomeFilter = button.dataset.outcome;
        document.querySelectorAll("[data-outcome]").forEach((item) => {
          item.setAttribute("aria-pressed", item === button ? "true" : "false");
        });
        listCap = LIST_STEP;
        paintList();
      });
    });
    const classPick = document.querySelector("[data-class-filter]");
    if (classPick) classPick.addEventListener("change", () => {
      classFilter = classPick.value;
      listCap = LIST_STEP;
      paintList();
    });
    const confidencePick = document.querySelector("[data-min-confidence]");
    if (confidencePick) confidencePick.addEventListener("change", () => {
      const number = Number(confidencePick.value);
      minConfidence = Number.isFinite(number) ? Math.min(1, Math.max(0, number)) : 0;
      confidencePick.value = String(minConfidence);
      listCap = LIST_STEP;
      paintList();
    });
    const equalPick = document.querySelector("[data-equal-classes]");
    if (equalPick) equalPick.addEventListener("change", () => {
      equalClasses = equalPick.checked;
      listCap = LIST_STEP;
      paintList();
    });
    const metalPick = document.querySelector("[data-hide-metal]");
    if (metalPick) metalPick.addEventListener("change", () => {
      hideMetal = metalPick.checked;
      listCap = LIST_STEP;
      paintList();
    });
    const distancePick = document.querySelector("[data-show-distance]");
    if (distancePick) distancePick.addEventListener("change", () => {
      showDistance = distancePick.checked;
      paintList();
    });
    const capPick = document.querySelector("[data-list-cap]");
    if (capPick) capPick.addEventListener("change", () => {
      const number = Number(capPick.value);
      listCap = Number.isFinite(number) ? Math.min(500, Math.max(12, Math.round(number))) : LIST_STEP;
      capPick.value = String(listCap);
      paintList();
    });
    if (scoreForm) scoreForm.addEventListener("submit", (event) => onScore(event));
    inspector.addEventListener("focusout", () => {
      setTimeout(() => {
        if (!inspectorStale || inspectorSelectOpen()) return;
        paintInspector();
      }, 0);
    });
    dialog.querySelector("[data-view-close]").addEventListener("click", () => dialog.close());
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        pending = null;
        paintPending();
      }
      if (event.key === "Backspace" || event.key === "Delete") {
        const tag = document.activeElement && document.activeElement.tagName;
        if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
        if (selectedId) removeNode(selectedId);
      }
    });
    board.addEventListener("pointermove", onDrag);
    board.addEventListener("pointerup", endDrag);
  }

  function viewSpot() {
    return {
      x: board.scrollLeft + 32,
      y: board.scrollTop + 32,
    };
  }

  function addNode(type, x, y) {
    const count = graph.nodes.length;
    const spot = viewSpot();
    const node = {
      id: nextId("n"),
      type,
      x: Number.isFinite(x) ? Math.max(0, x) : spot.x + (count % 4) * 28,
      y: Number.isFinite(y) ? Math.max(0, y) : spot.y + (count % 5) * 24,
    };
    if (type === "input" && graph.nodes.some((item) => item.type === "input")) {
      flowNote.textContent = "There is already an input. The split buttons choose which photos enter it.";
      return;
    }
    if (type === "belt") {
      const info = catalog[0];
      node.check = info.id;
      node.cut = info.cut;
      node.emptyWhen = info.empty_when;
    }
    if (type === "model") {
      node.model = "";
      node.confidence = 0.85;
      node.mode = "cloud";
      node.distance = 0.5;
      node.keepOn = false;
      node.keep = 0.95;
      node.clash = false;
    }
    if (type === "merge") node.title = "Merge";
    if (type === "sink") {
      node.name = "Pile";
      node.role = "hold";
    }
    graph.nodes.push(node);
    selectedId = node.id;
    pending = null;
    saveDraft();
    paintBoard();
    paintInspector();
    run();
  }

  function removeNode(id) {
    graph.nodes = graph.nodes.filter((node) => node.id !== id);
    graph.wires = graph.wires.filter((wire) => wire.from !== id && wire.to !== id);
    if (selectedId === id) selectedId = graph.nodes[0] ? graph.nodes[0].id : "";
    pending = null;
    saveDraft();
    paintBoard();
    paintInspector();
    run();
  }

  function tidy() {
    const depth = {};
    graph.nodes.forEach((node) => {
      depth[node.id] = 0;
    });
    for (let pass = 0; pass < graph.nodes.length; pass += 1) {
      graph.wires.forEach((wire) => {
        if (depth[wire.from] === undefined || depth[wire.to] === undefined) return;
        depth[wire.to] = Math.max(depth[wire.to], depth[wire.from] + 1);
      });
    }
    const columns = {};
    graph.nodes.forEach((node) => {
      const layer = depth[node.id] || 0;
      if (!columns[layer]) columns[layer] = [];
      columns[layer].push(node);
    });
    Object.keys(columns).sort((a, b) => Number(a) - Number(b)).forEach((layer) => {
      columns[layer].forEach((node, index) => {
        node.x = 24 + Number(layer) * 290;
        node.y = 24 + index * 168;
      });
    });
  }

  function paintBoard() {
    resizeCanvas();
    const cards = canvas.querySelectorAll(".flow-card");
    cards.forEach((card) => card.remove());
    graph.nodes.forEach((node) => canvas.appendChild(cardElement(node)));
    paintWires();
    paintPending();
  }

  function resizeCanvas() {
    let width = 980;
    let height = 680;
    graph.nodes.forEach((node) => {
      width = Math.max(width, node.x + 300);
      height = Math.max(height, node.y + 220);
    });
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
  }

  function cardElement(node) {
    const card = document.createElement("article");
    card.className = "flow-card"
      + (node.id === selectedId ? " selected" : "")
      + (viewNodes.indexOf(node.id) !== -1 ? " in-view" : "");
    card.dataset.node = node.id;
    card.style.left = node.x + "px";
    card.style.top = node.y + "px";
    const header = document.createElement("header");
    header.dataset.drag = "yes";
    const heading = document.createElement("strong");
    heading.textContent = titleOf(node);
    const sub = document.createElement("span");
    sub.textContent = subtitle(node);
    header.append(heading, sub);
    header.addEventListener("pointerdown", (event) => startDrag(event, node, card, header));
    header.addEventListener("click", () => selectNode(node.id));
    card.appendChild(header);
    if (node.type !== "input") {
      const inlet = document.createElement("button");
      inlet.type = "button";
      inlet.className = "flow-in";
      inlet.dataset.in = node.id;
      inlet.textContent = "in";
      inlet.addEventListener("click", (event) => {
        event.stopPropagation();
        connectTo(node.id);
      });
      card.appendChild(inlet);
    }
    const ports = document.createElement("div");
    ports.className = "flow-ports";
    portsOf(node).forEach((port) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "flow-port";
      button.dataset.port = port;
      const label = document.createElement("span");
      label.textContent = port;
      const count = document.createElement("span");
      count.dataset.portCount = node.id + ":" + port;
      count.textContent = "0";
      button.append(label, count);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        armPort(node.id, port);
      });
      ports.appendChild(button);
    });
    card.appendChild(ports);
    return card;
  }

  function subtitle(node) {
    if (node.type === "input") return "Photos in the split on screen";
    if (node.type === "belt") {
      const way = node.emptyWhen === "above" ? "empty above" : "empty below";
      return way + " " + formatCut(node.cut);
    }
    if (node.type === "model") {
      if (!node.model) return "No checkpoint yet";
      if (!modelReady(node.model)) return "Not scored yet";
      return modeLabel(node.mode) + " · " + Number(node.confidence).toFixed(2);
    }
    if (node.type === "merge") return "Joins every wire in";
    return node.role || "hold";
  }

  function selectNode(id) {
    selectedId = id;
    canvas.querySelectorAll(".flow-card").forEach((card) => {
      card.classList.toggle("selected", card.dataset.node === id);
    });
    paintInspector();
  }

  function armPort(nodeId, port) {
    if (pending && pending.from === nodeId && pending.port === port) {
      portFilter = portFilter && portFilter.nodeId === nodeId && portFilter.port === port
        ? null
        : { nodeId, port };
      pending = null;
      listCap = LIST_STEP;
      paintPending();
      paintList();
      return;
    }
    pending = { from: nodeId, port };
    portFilter = { nodeId, port };
    listCap = LIST_STEP;
    paintPending();
    paintList();
    const node = nodeById(nodeId);
    flowNote.textContent = "Wiring " + titleOf(node) + " · " + port + ". Click the next card’s input.";
  }

  function connectTo(targetId) {
    if (!pending) {
      selectNode(targetId);
      return;
    }
    if (pending.from === targetId || reaches(targetId, pending.from)) {
      flowNote.textContent = "That wire would loop. A later card can join a pile, but it cannot go back.";
      return;
    }
    const exists = graph.wires.some((wire) => (
      wire.from === pending.from && wire.port === pending.port && wire.to === targetId
    ));
    if (!exists) {
      graph.wires.push({
        id: nextId("w"),
        from: pending.from,
        port: pending.port,
        to: targetId,
      });
    }
    pending = null;
    saveDraft();
    paintWires();
    paintPending();
    run();
    flowNote.textContent = "Click an output, then the next card’s input. Click a wire to remove it.";
  }

  function reaches(fromId, targetId) {
    const seen = new Set();
    const stack = [fromId];
    while (stack.length) {
      const id = stack.pop();
      if (id === targetId) return true;
      if (seen.has(id)) continue;
      seen.add(id);
      graph.wires.forEach((wire) => {
        if (wire.from === id) stack.push(wire.to);
      });
    }
    return false;
  }

  function paintPending() {
    canvas.querySelectorAll(".flow-port").forEach((button) => {
      const card = button.closest("[data-node]");
      const on = pending && card && card.dataset.node === pending.from && button.dataset.port === pending.port;
      button.classList.toggle("armed", Boolean(on));
    });
  }

  function startDrag(event, node, card, header) {
    if (event.button !== undefined && event.button !== 0) return;
    event.preventDefault();
    drag = {
      id: node.id,
      pointer: event.pointerId,
      x: node.x,
      y: node.y,
      cx: event.clientX,
      cy: event.clientY,
      shift: event.shiftKey,
      moved: false,
    };
    board.style.overflow = "hidden";
    header.setPointerCapture(event.pointerId);
  }

  function onDrag(event) {
    if (!drag || event.pointerId !== drag.pointer) return;
    if (Math.hypot(event.clientX - drag.cx, event.clientY - drag.cy) > 4) drag.moved = true;
    event.preventDefault();
    const node = nodeById(drag.id);
    const card = canvas.querySelector('[data-node="' + drag.id + '"]');
    if (!node || !card) return;
    node.x = Math.max(0, drag.x + event.clientX - drag.cx);
    node.y = Math.max(0, drag.y + event.clientY - drag.cy);
    card.style.left = node.x + "px";
    card.style.top = node.y + "px";
    const left = board.scrollLeft;
    const top = board.scrollTop;
    resizeCanvas();
    board.scrollLeft = left;
    board.scrollTop = top;
    paintWires();
  }

  function endDrag() {
    if (!drag) return;
    const node = nodeById(drag.id);
    const moved = drag.moved;
    const shift = drag.shift;
    const id = drag.id;
    drag = null;
    board.style.overflow = "";
    if (node) {
      revealNode(node);
      selectNode(node.id);
      if (!moved) chooseBlocks(id, shift);
    }
    saveDraft();
  }

  function revealNode(node) {
    const margin = 24;
    const right = node.x + 240;
    const bottom = node.y + 180;
    if (node.x < board.scrollLeft + margin) board.scrollLeft = Math.max(0, node.x - margin);
    else if (right > board.scrollLeft + board.clientWidth - margin) {
      board.scrollLeft = right - board.clientWidth + margin;
    }
    if (node.y < board.scrollTop + margin) board.scrollTop = Math.max(0, node.y - margin);
    else if (bottom > board.scrollTop + board.clientHeight - margin) {
      board.scrollTop = bottom - board.clientHeight + margin;
    }
  }

  function paintWires() {
    svg.replaceChildren();
    graph.wires.forEach((wire) => {
      const from = canvas.querySelector('[data-node="' + wire.from + '"] [data-port="' + wire.port + '"]');
      const inlet = canvas.querySelector('[data-in="' + wire.to + '"]');
      if (!from || !inlet) return;
      const start = pointOf(from, "right");
      const end = pointOf(inlet, "left");
      const bend = Math.max(48, (end.x - start.x) / 2);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", "M " + start.x + " " + start.y + " C " + (start.x + bend) + " " + start.y + ", " + (end.x - bend) + " " + end.y + ", " + end.x + " " + end.y);
      path.dataset.wire = wire.id;
      path.addEventListener("click", () => {
        graph.wires = graph.wires.filter((item) => item.id !== wire.id);
        pending = null;
        saveDraft();
        paintWires();
        run();
      });
      svg.appendChild(path);
    });
    paintPortCounts();
  }

  function pointOf(element, side) {
    const origin = canvas.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    return {
      x: (side === "right" ? rect.right : rect.left) - origin.left,
      y: rect.top + rect.height / 2 - origin.top,
    };
  }

  function inspectorSelectOpen() {
    const active = document.activeElement;
    return Boolean(active && active.tagName === "SELECT" && inspector.contains(active));
  }

  function paintInspector() {
    if (inspectorSelectOpen()) {
      inspectorStale = true;
      return;
    }
    inspectorStale = false;
    inspector.replaceChildren();
    const node = nodeById(selectedId);
    if (!node) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "Select a card.";
      inspector.appendChild(empty);
      return;
    }
    const heading = document.createElement("h2");
    heading.textContent = titleOf(node);
    inspector.appendChild(heading);
    if (node.type === "input") {
      inspector.appendChild(hint("Every photo in the split on screen enters here."));
    }
    if (node.type === "belt") paintBeltInspector(node);
    if (node.type === "model") paintModelInspector(node);
    if (node.type === "merge") {
      inspector.appendChild(field("Name", "text", node.title || "", (value) => {
        node.title = value || "Merge";
        refreshCard(node);
      }));
    }
    if (node.type === "sink") {
      inspector.appendChild(field("Name", "text", node.name || "", (value) => {
        node.name = value || "Pile";
        refreshCard(node);
      }));
      inspector.appendChild(selectField("Role", sinkRoles(), node.role || "hold", (value) => {
        node.role = value;
        refreshCard(node);
        run();
      }));
      inspector.appendChild(hint("Metal and plastic piles are sorted. Empty is not a hold. Hold, and any unwired port, count toward the 5% cap when you fit."));
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ghost";
    remove.textContent = "Remove card";
    remove.addEventListener("click", () => removeNode(node.id));
    inspector.appendChild(remove);
  }

  function paintBeltInspector(node) {
    const picker = document.createElement("label");
    picker.textContent = "Check";
    const select = document.createElement("select");
    catalog.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.label;
      if (item.id === node.check) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      const info = checkInfo(select.value);
      node.check = info.id;
      node.cut = info.cut;
      node.emptyWhen = info.empty_when;
      saveDraft();
      paintInspector();
      refreshCard(node);
      run();
    });
    picker.appendChild(select);
    inspector.appendChild(picker);
    inspector.appendChild(hint(checkInfo(node.check).detail));
    inspector.appendChild(field("Cut", "number", node.cut, (value) => {
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      node.cut = number;
      refreshCard(node);
      run();
    }));
    const flip = document.createElement("label");
    flip.className = "check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = node.emptyWhen === "above";
    box.addEventListener("change", () => {
      node.emptyWhen = box.checked ? "above" : "below";
      refreshCard(node);
      run();
    });
    flip.append(box, document.createTextNode(" Empty when the score is above the cut"));
    inspector.appendChild(flip);
  }

  function paintModelInspector(node) {
    const picker = document.createElement("label");
    picker.textContent = "Checkpoint";
    const select = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Pick a checkpoint";
    select.appendChild(blank);
    modelChoices.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.rel;
      option.textContent = item.label;
      if (item.rel === node.model) option.selected = true;
      select.appendChild(option);
    });
    select.value = node.model || "";
    select.addEventListener("change", () => {
      const value = select.value;
      if (!select.isConnected || value === (node.model || "")) return;
      node.model = value;
      pruneWires();
      saveDraft();
      repaintCard(node);
      ensureModels();
    });
    picker.appendChild(select);
    inspector.appendChild(picker);
    if (node.model && !modelReady(node.model)) {
      inspector.appendChild(hint("Not scored on this dataset yet. Score this checkpoint reads that model only and leaves the belt cache."));
    } else if (node.model && needsResinRefresh(node.model)) {
      inspector.appendChild(hint("This cache has no HDPE, PET, or PP call. Rescore to route those outputs."));
    }
    inspector.appendChild(field("Confidence", "number", node.confidence, (value) => {
      node.confidence = snap(value, 0.5, 0.99, 0.85);
      refreshCard(node);
      run();
    }, "0.01"));
    const useDistance = document.createElement("label");
    useDistance.className = "check";
    const distanceBox = document.createElement("input");
    distanceBox.type = "checkbox";
    distanceBox.checked = node.distanceOn !== false;
    distanceBox.addEventListener("change", () => {
      node.distanceOn = distanceBox.checked;
      saveDraft();
      paintInspector();
      run();
    });
    useDistance.append(distanceBox, document.createTextNode(" Use distance"));
    inspector.appendChild(useDistance);
    if (node.distanceOn !== false) {
      inspector.appendChild(selectField("Distance", MODES.map((item) => item[0]), node.mode, (value) => {
        node.mode = value;
        refreshCard(node);
        run();
      }, MODES.map((item) => item[1])));
      inspector.appendChild(field("Mark", "number", node.distance, (value) => {
        const number = Number(value);
        if (!Number.isFinite(number) || number < 0) return;
        node.distance = number;
        run();
      }, "any"));
    }
    const keep = document.createElement("label");
    keep.className = "check";
    const keepBox = document.createElement("input");
    keepBox.type = "checkbox";
    keepBox.checked = Boolean(node.keepOn);
    keepBox.addEventListener("change", () => {
      node.keepOn = keepBox.checked;
      run();
    });
    keep.append(keepBox, document.createTextNode(" Keep once confidence reaches this, whatever the distance"));
    inspector.appendChild(keep);
    inspector.appendChild(field("Keep", "number", node.keep, (value) => {
      node.keep = snap(value, 0.5, 0.999, 0.95);
      run();
    }, "0.001"));
    const clash = document.createElement("label");
    clash.className = "check";
    const clashBox = document.createElement("input");
    clashBox.type = "checkbox";
    clashBox.checked = Boolean(node.clash);
    clashBox.addEventListener("change", () => {
      node.clash = clashBox.checked;
      run();
    });
    clash.append(clashBox, document.createTextNode(" Also mark others when the nearest cloud is the other class"));
    inspector.appendChild(clash);
    const actions = document.createElement("div");
    actions.className = "buttons";
    const useMark = document.createElement("button");
    useMark.type = "button";
    useMark.className = "ghost";
    useMark.textContent = "Validation 99%";
    useMark.addEventListener("click", () => {
      const payload = loaded.get(node.model);
      const mark = payload && payload.suggestions ? Number(payload.suggestions[node.mode]) : NaN;
      if (!Number.isFinite(mark)) return;
      node.distance = mark;
      saveDraft();
      paintInspector();
      run();
    });
    actions.appendChild(useMark);
    if (node.model) {
      const scoreOne = document.createElement("button");
      scoreOne.type = "button";
      scoreOne.textContent = modelReady(node.model) ? "Rescore this checkpoint" : "Score this checkpoint";
      scoreOne.addEventListener("click", () => scoreOneModel(node));
      actions.appendChild(scoreOne);
    }
    inspector.appendChild(actions);
    const goal = document.createElement("label");
    goal.textContent = "Fit for";
    const goalSelect = document.createElement("select");
    goalSelect.dataset.fitGoal = "yes";
    [
      ["balanced", "Balanced accuracy"],
      ["accuracy", "Accuracy on accepted"],
      ["still", "Fewest still wrong"],
      ["tradeoff", "Catch misses, hold fewer correct"],
    ].forEach((item) => {
      const option = document.createElement("option");
      option.value = item[0];
      option.textContent = item[1];
      goalSelect.appendChild(option);
    });
    goal.appendChild(goalSelect);
    inspector.appendChild(goal);
    const fit = document.createElement("button");
    fit.type = "button";
    fit.textContent = "Fit on validation";
    fit.addEventListener("click", () => fitNode(node, goalSelect.value, fitNote));
    inspector.appendChild(fit);
    const fitNote = document.createElement("p");
    fitNote.className = "hint";
    fitNote.textContent = "Manual. Fit searches confidence, the four distances, and the keep line on validation photos that reach this card. Test does not pick it.";
    inspector.appendChild(fitNote);
  }

  function hint(text) {
    const node = document.createElement("p");
    node.className = "hint";
    node.textContent = text;
    return node;
  }

  function field(label, type, value, onChange, step) {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    const input = document.createElement("input");
    input.type = type;
    input.value = value;
    if (step) input.step = step;
    input.addEventListener("change", () => {
      onChange(input.value);
      saveDraft();
    });
    wrap.appendChild(input);
    return wrap;
  }

  function selectField(label, values, current, onChange, texts) {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    const select = document.createElement("select");
    values.forEach((value, index) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = texts ? texts[index] : value;
      if (value === current) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      onChange(select.value);
      saveDraft();
    });
    wrap.appendChild(select);
    return wrap;
  }

  function repaintCard(node) {
    const card = canvas.querySelector('[data-node="' + node.id + '"]');
    const next = cardElement(node);
    if (card) card.replaceWith(next);
    else canvas.appendChild(next);
    paintWires();
    paintPending();
  }

  function refreshCard(node) {
    const card = canvas.querySelector('[data-node="' + node.id + '"]');
    if (!card) return;
    const heading = card.querySelector("strong");
    const sub = card.querySelector("header span");
    if (heading) heading.textContent = titleOf(node);
    if (sub) sub.textContent = subtitle(node);
  }

  function snap(value, low, high, fallback) {
    let number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    if (number > 1 && high <= 1) number /= 100;
    return Math.min(high, Math.max(low, number));
  }

  function modeLabel(mode) {
    const found = MODES.find((item) => item[0] === mode);
    return found ? found[1] : "distance";
  }

  function modelReady(rel) {
    const payload = loaded.get(rel);
    return Boolean(payload && payload.ready && payload.frames && payload.frames.length);
  }

  function needsResinRefresh(rel) {
    if (modelClasses(rel).indexOf("HDPE") === -1) return false;
    const payload = loaded.get(rel);
    if (!payload || !Array.isArray(payload.frames) || !payload.frames.length) return false;
    const bank = payload.bank_classes || [];
    if (bank.indexOf("HDPE") === -1 || bank.indexOf("PET") === -1 || bank.indexOf("PP") === -1) return true;
    return !payload.frames.some((frame) => frame.pred === "HDPE" || frame.pred === "PET" || frame.pred === "PP");
  }

  function sinkRoles() {
    const roles = [];
    graph.nodes.forEach((node) => {
      if (node.type !== "model") return;
      modelClasses(node.model).forEach((name) => {
        if (roles.indexOf(name) === -1) roles.push(name);
      });
    });
    ["metal", "plastic", "empty", "hold"].forEach((name) => {
      if (roles.indexOf(name) === -1) roles.push(name);
    });
    return roles;
  }

  function pruneWires() {
    const before = graph.wires.length;
    graph.wires = graph.wires.filter((wire) => {
      const node = nodeById(wire.from);
      return node && portsOf(node).indexOf(wire.port) !== -1;
    });
    return graph.wires.length !== before;
  }

  function trueResinOf(frame) {
    if (!frame) return "";
    if (frame.trueResin) return frame.trueResin;
    return resinLib.fromPath(frame.path);
  }

  function callMatchesLabel(call, frame) {
    if (!call || !frame) return false;
    const resin = trueResinOf(frame);
    const resins = ["HDPE", "PET", "PP"];
    if (resins.indexOf(call) !== -1) {
      if (resins.indexOf(resin) !== -1) return call === resin;
      if (resin === "OTHERS" || frame.true === "metal") return false;
      if (frame.true === "plastic") return true;
      return false;
    }
    if (call === frame.true) return true;
    if (resin && call === resin) return true;
    if (call === "plastic" && resins.indexOf(resin) !== -1) return true;
    if (call === "metal" && resin === "OTHERS") return true;
    return false;
  }

  function knownFrame(frame) {
    if (!frame) return false;
    if (frame.true === "metal" || frame.true === "plastic") return true;
    return resinLib.order.indexOf(trueResinOf(frame)) !== -1;
  }

  function truthKey(frame) {
    const resin = trueResinOf(frame);
    const resinSink = graph.nodes.some((node) => (
      node.type === "sink" && resinLib.order.indexOf(node.role) !== -1
    ));
    if (resinSink && resin) return resin;
    return frame.true || resin || "";
  }

  function sortRole(role) {
    return Boolean(role) && role !== "empty" && role !== "hold";
  }

  function preferRole(current, next, frame) {
    if (!current) return next;
    const currentSort = sortRole(current);
    const nextSort = sortRole(next);
    if (currentSort && callMatchesLabel(current, frame)) return current;
    if (nextSort && callMatchesLabel(next, frame)) return next;
    if (currentSort) return current;
    if (nextSort) return next;
    if (current === "empty" || next === "empty") return "empty";
    return current || next;
  }

  async function fetchJson(url) {
    const target = new URL(url, window.location.origin);
    target.searchParams.set("_", String(Date.now()));
    const response = await fetch(target, { cache: "no-store" });
    if (!response.ok) throw new Error("load");
    return response.json();
  }

  async function loadBelt() {
    try {
      belt = await fetchJson(root.dataset.beltUrl);
    } catch (error) {
      flowNote.textContent = "Could not load the belt cache.";
      belt = { ready: false, frames: [] };
    }
    if (belt && Array.isArray(belt.defaults) && belt.defaults.length) catalog = belt.defaults;
    joinFrames();
    await ensureModels();
  }

  async function ensureModels() {
    const rels = [];
    graph.nodes.forEach((node) => {
      if (node.type === "model" && node.model && !loaded.has(node.model) && rels.indexOf(node.model) === -1) {
        rels.push(node.model);
      }
    });
    await Promise.all(rels.map(async (rel) => {
      try {
        const url = new URL(root.dataset.modelUrl, window.location.origin);
        url.searchParams.set("model", rel);
        loaded.set(rel, await fetchJson(url));
      } catch (error) {
        loaded.set(rel, { ready: false, model: rel, frames: [] });
      }
    }));
    joinFrames();
    if (pruneWires()) saveDraft();
    paintBoard();
    paintInspector();
    run();
    paintScoreButton();
  }

  async function refreshScores() {
    loaded.clear();
    try {
      belt = await fetchJson(root.dataset.beltUrl);
    } catch (error) {
      belt = { ready: false, frames: [] };
    }
    root.dataset.beltReady = belt && belt.ready ? "yes" : "no";
    if (belt && Array.isArray(belt.defaults) && belt.defaults.length) catalog = belt.defaults;
    await ensureModels();
    modelRels().forEach((rel) => {
      if (modelReady(rel)) scoredAtLoad.add(rel);
    });
    paintScoreButton();
  }

  function joinFrames() {
    const map = new Map();
    const beltFrames = belt && Array.isArray(belt.frames) ? belt.frames : [];
    beltFrames.forEach((frame) => {
      map.set(frame.path, {
        path: frame.path,
        split: frame.split,
        true: frame.true || "",
        trueResin: frame.resin || resinLib.fromPath(frame.path),
        scores: frame,
        models: {},
      });
    });
    loaded.forEach((payload, rel) => {
      if (!payload || !payload.ready) return;
      (payload.frames || []).forEach((frame) => {
        let row = map.get(frame.path);
        if (!row) {
          row = {
            path: frame.path,
            split: frame.split,
            true: frame.true || "",
            trueResin: resinLib.fromPath(frame.path),
            scores: {},
            models: {},
          };
          map.set(frame.path, row);
        }
        row.models[rel] = frame;
        if (!row.true && frame.true) row.true = frame.true;
      });
    });
    frames = Array.from(map.values());
  }

  function visibleFrames() {
    if (split === "both") {
      return frames.filter((frame) => frame.split === "validation" || frame.split === "test");
    }
    return frames.filter((frame) => frame.split === split);
  }

  function run() {
    const rows = visibleFrames();
    outcome = evaluate(graph, rows, true);
    paintPortCounts();
    paintFigures(rows, outcome);
    paintList();
    paintRecipe();
    const notes = {
      validation: "Validation is the set a fit is allowed to use.",
      test: "Test was scored into the cache. It does not pick a cut.",
      both: "Validation and test together. A fit still uses validation only.",
      probe: "Probes, including empty belts. A fit does not use them.",
      garbage: "The Garbage folder. A fit does not use it.",
    };
    if (splitNote) splitNote.textContent = notes[split] || "";
  }

  function evaluate(current, rows, record) {
    const byId = new Map(current.nodes.map((node) => [node.id, node]));
    const order = topo(current.nodes, current.wires);
    const count = rows.length;
    const result = {
      order,
      ports: {},
      arrived: {},
      visits: record ? rows.map(() => []) : null,
      roles: rows.map(() => ""),
      holdMark: new Uint8Array(count),
      dropped: new Uint8Array(count),
      lastPred: rows.map(() => ""),
    };
    if (!order) return result;
    const arrived = {};
    current.nodes.forEach((node) => {
      arrived[node.id] = [];
    });
    const everyone = [];
    for (let index = 0; index < count; index += 1) everyone.push(index);
    order.forEach((id) => {
      const node = byId.get(id);
      const incoming = node.type === "input" ? everyone : unique(arrived[id]);
      result.arrived[id] = incoming;
      if (node.type === "sink") {
        const role = node.role || "hold";
        incoming.forEach((index) => {
          result.roles[index] = preferRole(result.roles[index], role, rows[index]);
          if (role === "hold") result.holdMark[index] = 1;
          if (record) result.visits[index].push({ nodeId: id, port: "in" });
        });
        result.ports[id + ":in"] = incoming;
        return;
      }
      const groups = classify(node, incoming, rows, result);
      Object.keys(groups).forEach((port) => {
        const members = groups[port];
        result.ports[id + ":" + port] = members;
        const wires = current.wires.filter((wire) => wire.from === id && wire.port === port);
        members.forEach((index) => {
          if (record) result.visits[index].push({ nodeId: id, port });
          if (!wires.length) result.dropped[index] = 1;
          wires.forEach((wire) => arrived[wire.to].push(index));
        });
      });
    });
    return result;
  }

  function classify(node, incoming, rows, result) {
    if (node.type === "input" || node.type === "merge") return { out: incoming };
    if (node.type === "belt") {
      const groups = { occupied: [], empty: [] };
      incoming.forEach((index) => {
        const value = Number(rows[index].scores ? rows[index].scores[node.check] : NaN);
        if (!Number.isFinite(value)) {
          result.dropped[index] = 1;
          return;
        }
        const empty = node.emptyWhen === "above" ? value > Number(node.cut) : value < Number(node.cut);
        groups[empty ? "empty" : "occupied"].push(index);
      });
      return groups;
    }
    if (node.type === "model") {
      const groups = {};
      portsOf(node).forEach((port) => {
        groups[port] = [];
      });
      const payload = node.model ? rows : null;
      incoming.forEach((index) => {
        const scores = payload && rows[index].models ? rows[index].models[node.model] : null;
        if (!scores) {
          result.dropped[index] = 1;
          return;
        }
        const decision = judge(scores, node);
        result.lastPred[index] = predictedClass(scores, node);
        const port = groups[decision] ? decision : "others";
        groups[port].push(index);
      });
      return groups;
    }
    return {};
  }

  function predictedClass(frame, node) {
    const classes = modelClasses(node.model);
    const resin = frame.resin || "";
    if (resin && classes.indexOf(resin) !== -1) return resin;
    const pred = frame.pred || "";
    if (pred && classes.indexOf(pred) !== -1) return pred;
    if (resin) return resin;
    return pred || classes[0] || "metal";
  }

  function judge(frame, node) {
    const confidence = Number(frame.confidence);
    const distance = Number(frame[node.mode]);
    const far = node.distanceOn !== false && Number.isFinite(distance) && distance > Number(node.distance);
    const clash = node.clash && frame.disagree;
    const call = predictedClass(frame, node);
    if (node.keepOn && confidence >= Number(node.keep)) return call;
    if (far || clash) return "others";
    if (confidence < Number(node.confidence)) return "uncertain";
    return call;
  }

  function topo(nodes, wires) {
    const indegree = {};
    nodes.forEach((node) => {
      indegree[node.id] = 0;
    });
    wires.forEach((wire) => {
      if (indegree[wire.to] !== undefined) indegree[wire.to] += 1;
    });
    const queue = nodes.filter((node) => indegree[node.id] === 0).map((node) => node.id);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      wires.forEach((wire) => {
        if (wire.from !== id || indegree[wire.to] === undefined) return;
        indegree[wire.to] -= 1;
        if (indegree[wire.to] === 0) queue.push(wire.to);
      });
    }
    return order.length === nodes.length ? order : null;
  }

  function unique(list) {
    return Array.from(new Set(list));
  }

  function endedAs(index) {
    if (!outcome || !outcome.roles) return "dropped";
    return outcome.roles[index] || "dropped";
  }

  function summarize(rows, result) {
    const stats = {
      n: rows.length,
      accepted: 0,
      correct: 0,
      still: 0,
      held: 0,
      empty: 0,
      holdCapCount: 0,
      heldRight: 0,
      caught: 0,
      per: {},
      blankN: 0,
      blankCaught: 0,
      balanced: null,
      holdCap: 1,
    };
    if (!result || !result.order) return stats;
    rows.forEach((frame, index) => {
      const dropped = result.dropped[index];
      const role = result.roles[index] || (dropped ? "dropped" : "dropped");
      const known = knownFrame(frame);
      if (result.holdMark[index] || dropped) stats.holdCapCount += 1;
      if (frame.true === "empty") {
        stats.blankN += 1;
        if (role === "empty") stats.blankCaught += 1;
      }
      if (role === "empty") stats.empty += 1;
      if (role === "hold" || role === "dropped") stats.held += 1;
      if (sortRole(role)) {
        stats.accepted += 1;
        const hit = callMatchesLabel(role, frame);
        if (hit) stats.correct += 1;
        else if (known) stats.still += 1;
        const key = truthKey(frame);
        if (key) {
          if (!stats.per[key]) stats.per[key] = { n: 0, hit: 0 };
          stats.per[key].n += 1;
          if (hit) stats.per[key].hit += 1;
        }
      } else if (known) {
        const pred = result.lastPred[index];
        if (pred && callMatchesLabel(pred, frame)) stats.heldRight += 1;
        else if (pred) stats.caught += 1;
      }
    });
    const recalls = Object.keys(stats.per)
      .filter((key) => stats.per[key].n)
      .map((key) => stats.per[key].hit / stats.per[key].n);
    stats.balanced = recalls.length
      ? recalls.reduce((sum, value) => sum + value, 0) / recalls.length
      : null;
    stats.holdCap = stats.n ? stats.holdCapCount / stats.n : 1;
    return stats;
  }

  function paintPortCounts() {
    canvas.querySelectorAll("[data-port-count]").forEach((node) => {
      const members = outcome && outcome.ports ? outcome.ports[node.dataset.portCount] : null;
      node.textContent = members ? String(members.length) : "0";
    });
  }

  function paintFigures(rows, result) {
    if (!figures) return;
    figures.replaceChildren();
    if (!rows.length) {
      figures.appendChild(figure("Photos", "0", belt && belt.ready ? "None in this split" : "Score the belt first"));
      return;
    }
    if (!result.order) {
      figures.appendChild(figure("Loop", "—", "A wire goes back. Remove it."));
      return;
    }
    const stats = summarize(rows, result);
    figures.appendChild(figure("Photos", String(stats.n), "in this split"));
    figures.appendChild(figure("Class pile", share(stats.accepted, stats.n), stats.correct + " match the label"));
    figures.appendChild(figure("Still wrong", share(stats.still, stats.n), stats.still + " in the other class pile"));
    figures.appendChild(figure("Held", share(stats.held, stats.n), stats.held + " in Hold or dropped"));
    figures.appendChild(figure("Empty", share(stats.empty, stats.n), stats.empty + " in Empty"));
    if (stats.blankN) {
      figures.appendChild(figure("Blanks caught", share(stats.blankCaught, stats.blankN), stats.blankCaught + " of " + stats.blankN));
    }
  }

  function figure(label, value, sub) {
    const node = document.createElement("div");
    node.className = "metric";
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = value;
    const detail = document.createElement("span");
    detail.className = "sub";
    detail.textContent = sub;
    const name = document.createElement("span");
    name.className = "lbl";
    name.textContent = label;
    node.append(num, detail, name);
    return node;
  }

  function share(count, total) {
    if (!total) return "—";
    return (100 * count / total).toFixed(1) + "%";
  }

  function chooseBlocks(id, shift) {
    if (shift) {
      const index = viewNodes.indexOf(id);
      if (index === -1) viewNodes.push(id);
      else viewNodes.splice(index, 1);
    } else if (viewNodes.length === 1 && viewNodes[0] === id) {
      viewNodes = [];
    } else {
      viewNodes = [id];
    }
    listCap = LIST_STEP;
    canvas.querySelectorAll(".flow-card").forEach((card) => {
      card.classList.toggle("in-view", viewNodes.indexOf(card.dataset.node) !== -1);
    });
    paintList();
  }

  function stepsToShow() {
    const order = outcome && outcome.order ? outcome.order : [];
    const nodes = order.map((id) => nodeById(id)).filter(Boolean);
    if (!viewNodes.length) return nodes;
    return nodes.filter((node) => viewNodes.indexOf(node.id) !== -1);
  }

  function stepDecision(node, index) {
    if (!outcome || !outcome.visits || !outcome.visits[index]) return "";
    const visits = outcome.visits[index];
    for (let cursor = visits.length - 1; cursor >= 0; cursor -= 1) {
      if (visits[cursor].nodeId === node.id) return visits[cursor].port || "";
    }
    return "";
  }

  function reachedIndexes(node) {
    if (!outcome) return [];
    if (portFilter && portFilter.nodeId === node.id && outcome.ports) {
      return (outcome.ports[node.id + ":" + portFilter.port] || []).slice();
    }
    return (outcome.arrived && outcome.arrived[node.id] ? outcome.arrived[node.id] : []).slice();
  }

  function balanceKey(node, frame, index) {
    const decision = stepDecision(node, index);
    if (decision === "uncertain" || decision === "others") return decision;
    const resin = trueResinOf(frame);
    if (resin && resin !== "OTHERS") return resin;
    return frame.true || decision || "unknown";
  }

  function passesClass(node, frame, index) {
    if (classFilter === "any") return true;
    if (classFilter === "uncertain" || classFilter === "others") {
      return stepDecision(node, index) === classFilter;
    }
    const resin = trueResinOf(frame);
    if (classFilter === "plastic") {
      return frame.true === "plastic" || ["HDPE", "PET", "PP"].indexOf(resin) !== -1;
    }
    if (classFilter === "metal") return frame.true === "metal" || resin === "OTHERS";
    return resin === classFilter;
  }

  function stepIsRight(node, frame, index) {
    const decision = stepDecision(node, index);
    if (node.type === "belt") {
      if (frame.true === "empty") return decision === "empty";
      return decision !== "empty";
    }
    if (node.type === "sink") return callMatchesLabel(node.role || "", frame);
    if (node.type === "input" || node.type === "merge") return true;
    if (decision === "uncertain" || decision === "others") return false;
    return callMatchesLabel(decision, frame);
  }

  function passesOutcome(node, frame, index) {
    if (outcomeFilter === "all") return true;
    if (node.type === "input" || node.type === "merge") return true;
    const match = stepIsRight(node, frame, index);
    return outcomeFilter === "right" ? match : !match;
  }

  function passesConfidence(node, frame) {
    if (!(minConfidence > 0) || node.type !== "model") return true;
    const row = node.model && frame.models ? frame.models[node.model] : null;
    const value = row ? Number(row.confidence) : NaN;
    return Number.isFinite(value) && value >= minConfidence;
  }

  function filteredIndexes(node, rows) {
    const kept = [];
    reachedIndexes(node).forEach((index) => {
      const frame = rows[index];
      if (!frame) return;
      if (hideMetal && (frame.true === "metal" || trueResinOf(frame) === "OTHERS")) return;
      if (!passesClass(node, frame, index)) return;
      if (!passesOutcome(node, frame, index)) return;
      if (!passesConfidence(node, frame)) return;
      kept.push({ index, key: balanceKey(node, frame, index) });
    });
    if (!equalClasses) {
      const all = kept.map((item) => item.index);
      return {
        shown: all.slice(0, listCap),
        total: all.length,
        more: all.length > listCap,
        counts: portCounts(node, all),
      };
    }
    const groups = new Map();
    kept.forEach((item) => {
      if (!groups.has(item.key)) groups.set(item.key, []);
      groups.get(item.key).push(item.index);
    });
    let smallest = Infinity;
    groups.forEach((list) => {
      if (list.length < smallest) smallest = list.length;
    });
    if (!Number.isFinite(smallest)) return { shown: [], total: 0, more: false, counts: {} };
    const perClass = Math.min(smallest, listCap);
    const shown = [];
    const balanced = [];
    groups.forEach((list) => {
      list.slice(0, smallest).forEach((index) => balanced.push(index));
      list.slice(0, perClass).forEach((index) => shown.push(index));
    });
    return {
      shown,
      total: balanced.length,
      more: perClass < smallest,
      counts: portCounts(node, balanced),
    };
  }

  function portCounts(node, indexes) {
    const counts = {};
    indexes.forEach((index) => {
      const port = stepDecision(node, index) || "in";
      counts[port] = (counts[port] || 0) + 1;
    });
    return counts;
  }

  function paintList() {
    if (!stepsHost) return;
    const rows = visibleFrames();
    stepsHost.replaceChildren();
    if (!outcome || !outcome.order) {
      if (listNote) listNote.textContent = "The wires loop, so there is no path to list.";
      if (moreButton) moreButton.hidden = true;
      return;
    }
    const steps = stepsToShow();
    let truncated = false;
    let shownCount = 0;
    let totalCount = 0;
    const openIds = viewNodes.length ? viewNodes : [];
    steps.forEach((node) => {
      const batch = filteredIndexes(node, rows);
      const open = openIds.indexOf(node.id) !== -1;
      totalCount += batch.total;
      if (open && batch.more) truncated = true;
      if (open) shownCount += batch.shown.length;
      const section = document.createElement("section");
      section.className = "step-block";
      const bar = document.createElement("button");
      bar.type = "button";
      bar.className = "step-bar" + (open ? " open" : "");
      const bits = Object.keys(batch.counts || {}).map((key) => batch.counts[key] + " " + key);
      const portNote = portFilter && portFilter.nodeId === node.id ? " · " + portFilter.port : "";
      const countLine = (bits.join(" · ") || "No photos") + portNote
        + (open && batch.shown.length < batch.total ? ". Showing " + batch.shown.length + " of " + batch.total : "");
      const name = document.createElement("strong");
      name.textContent = titleOf(node);
      const detail = document.createElement("span");
      detail.textContent = countLine;
      bar.append(name, detail);
      bar.addEventListener("click", (event) => chooseBlocks(node.id, event.shiftKey));
      section.appendChild(bar);
      if (open) {
        const grid = document.createElement("div");
        grid.className = "grid";
        batch.shown.forEach((index) => grid.appendChild(photoButton(rows[index], index, node)));
        section.appendChild(grid);
      }
      stepsHost.appendChild(section);
    });
    if (listNote) {
      const where = viewNodes.length
        ? viewNodes.map((id) => titleOf(nodeById(id))).join(", ")
        : "Every step is collapsed";
      listNote.textContent = where + ". " + totalCount + " photos across these steps. "
        + (viewNodes.length && shownCount < totalCount ? "Showing " + shownCount + ". " : "")
        + "Green is right, red is wrong, amber is others, gray is uncertain or empty. The left edge is the resin. Click a block to open its photos.";
    }
    if (moreButton) moreButton.hidden = !truncated;
  }

  function photoButton(frame, index, node) {
    const button = document.createElement("button");
    button.type = "button";
    const resinName = trueResinOf(frame);
    button.className = "photo-hit tone-" + stepTone(node, frame, index);
    if (resinLib.colors[resinName]) button.dataset.resin = resinName;
    const image = document.createElement("img");
    image.alt = "";
    image.src = fileUrl(frame.path);
    const caption = document.createElement("p");
    caption.textContent = tileText(node, frame, index);
    button.title = caption.textContent;
    button.append(image, caption);
    button.addEventListener("click", () => openPhoto(frame, index));
    return button;
  }

  function tileText(node, frame, index) {
    const decision = stepDecision(node, index);
    if (node.type === "model") {
      const row = node.model && frame.models ? frame.models[node.model] : null;
      const predicted = row ? (row.resin || row.pred || decision) : (decision || "—");
      const confidence = row && Number.isFinite(Number(row.confidence))
        ? Number(row.confidence).toFixed(2)
        : "—";
      const distance = showDistance && row ? " · " + formatCut(row[node.mode]) : "";
      return trueLabel(frame) + " · " + predicted + " · " + confidence + distance;
    }
    if (node.type === "belt") {
      const value = frame.scores ? frame.scores[node.check] : null;
      return (decision || "—") + " · " + formatCut(value);
    }
    return (decision || node.role || "in") + " · " + trueLabel(frame);
  }

  function stepTone(node, frame, index) {
    const decision = stepDecision(node, index);
    if (decision === "others") return "warn";
    if (decision === "uncertain" || decision === "hold") return "hold";
    if (decision === "dropped") return "bad";
    if (node.type === "input" || node.type === "merge") return "hold";
    return stepIsRight(node, frame, index) ? "good" : "bad";
  }

  function openPhoto(frame, index) {
    dialog.querySelector("[data-view-title]").textContent = frame.path.split("/").pop();
    const image = dialog.querySelector("[data-view-image]");
    image.src = fileUrl(frame.path);
    image.alt = frame.path;
    dialog.querySelector("[data-view-caption]").textContent = frame.path;
    const detail = dialog.querySelector("[data-view-detail]");
    detail.replaceChildren();
    const role = outcome ? endedAs(index) : "dropped";
    detail.appendChild(hint("Label " + trueLabel(frame) + ". Ended as " + role + "."));
    const path = document.createElement("ol");
    path.className = "flow-path";
    if (outcome && outcome.visits) {
      outcome.visits[index].forEach((step) => {
        const item = document.createElement("li");
        item.textContent = titleOf(nodeById(step.nodeId)) + " · " + step.port;
        path.appendChild(item);
      });
    }
    detail.appendChild(path);
    const scores = frame.scores || {};
    catalog.forEach((item) => {
      if (scores[item.id] === undefined || scores[item.id] === null) return;
      detail.appendChild(hint(item.label + " " + formatCut(scores[item.id])));
    });
    Object.keys(frame.models || {}).forEach((rel) => {
      const row = frame.models[rel];
      detail.appendChild(hint(
        modelTitle(rel) + " " + (row.resin || row.pred || "—")
        + " " + Number(row.confidence).toFixed(3)
        + " · cloud " + formatCut(row.cloud)
        + " · call " + formatCut(row.call)
        + " · k-NN " + formatCut(row.knn)
        + " · Mahalanobis " + formatCut(row.maha)
      ));
    });
    if (!dialog.open) dialog.showModal();
  }

  function fileUrl(path) {
    return "/files/" + String(path).split("/").map(encodeURIComponent).join("/");
  }

  function paintRecipe() {
    if (!recipe) return;
    recipe.replaceChildren();
    const lines = recipeLines();
    lines.forEach((text) => {
      const paragraph = document.createElement("p");
      paragraph.textContent = text;
      recipe.appendChild(paragraph);
    });
  }

  function recipeLines() {
    const order = topo(graph.nodes, graph.wires);
    if (!order) return ["The wires loop, so this is not a pipeline yet. Remove the wire that goes back."];
    const lines = [];
    order.forEach((id) => {
      const node = nodeById(id);
      lines.push(describe(node));
      portsOf(node).forEach((port) => {
        const targets = graph.wires
          .filter((wire) => wire.from === id && wire.port === port)
          .map((wire) => titleOf(nodeById(wire.to)));
        const name = titleOf(node) + " · " + port;
        if (!targets.length) lines.push(name + " is unwired, so those photos are dropped.");
        else lines.push(name + " goes to " + targets.join(" and ") + ".");
      });
    });
    return lines;
  }

  function describe(node) {
    if (node.type === "input") return "Start with the photos in the split on screen.";
    if (node.type === "belt") {
      const info = checkInfo(node.check);
      const way = node.emptyWhen === "above" ? "above" : "below";
      return info.label + ": empty when the score is " + way + " " + formatCut(node.cut) + ". Otherwise occupied.";
    }
    if (node.type === "model") {
      if (!node.model) return titleOf(node) + " has no checkpoint, so photos that reach it are dropped.";
      const keep = node.keepOn
        ? " A photo at or above " + Number(node.keep).toFixed(3) + " confidence keeps the winning class."
        : "";
      return titleOf(node) + ": others when " + modeLabel(node.mode) + " is above " + formatCut(node.distance) + ". Uncertain when confidence is under " + Number(node.confidence).toFixed(2) + "." + keep;
    }
    if (node.type === "merge") return titleOf(node) + " joins every wire that comes in.";
    return titleOf(node) + " is the " + (node.role || "hold") + " pile.";
  }

  function formatCut(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const abs = Math.abs(number);
    if (abs >= 100) return String(Math.round(number));
    if (abs >= 10) return number.toFixed(1);
    if (abs >= 1) return number.toFixed(2);
    return number.toFixed(3);
  }

  function fitNode(node, goal, note) {
    if (!node.model || !modelReady(node.model)) {
      note.textContent = "Score this checkpoint before fitting it.";
      return;
    }
    const validation = frames.filter((frame) => frame.split === "validation");
    if (!validation.length) {
      note.textContent = "No validation photos in the cache.";
      return;
    }
    const saved = {
      confidence: node.confidence,
      mode: node.mode,
      distance: node.distance,
      keepOn: node.keepOn,
      keep: node.keep,
    };
    note.textContent = "Fitting on validation…";
    window.setTimeout(() => {
      const reached = new Set((evaluate(graph, validation, false).arrived[node.id]) || []);
      let best = null;
      let tried = 0;
      MODES.forEach((mode) => {
        const distances = distanceGrid(validation, reached, node, mode[0]);
        FLOORS.forEach((floor) => {
          const keeps = [{ on: false, value: floor }];
          KEEPS.forEach((value) => {
            if (value + 1e-9 >= floor) keeps.push({ on: true, value });
          });
          distances.forEach((distance) => {
            keeps.forEach((keep) => {
              node.confidence = floor;
              node.mode = mode[0];
              node.distance = distance;
              node.keepOn = keep.on;
              node.keep = keep.value;
              const stats = summarize(validation, evaluate(graph, validation, false));
              tried += 1;
              const candidate = {
                confidence: floor,
                mode: mode[0],
                distance,
                keepOn: keep.on,
                keep: keep.value,
                hold: stats.holdCap,
                key: goalKey(stats, goal),
              };
              if (betterCandidate(candidate, best)) best = candidate;
            });
          });
        });
      });
      const chosen = best && best.hold <= HOLD_CAP ? best : null;
      const applied = chosen || saved;
      node.confidence = applied.confidence;
      node.mode = applied.mode;
      node.distance = applied.distance;
      node.keepOn = applied.keepOn;
      node.keep = applied.keep;
      saveDraft();
      paintInspector();
      refreshCard(node);
      run();
      const notes = inspector.querySelectorAll(".hint");
      const target = notes.length ? notes[notes.length - 1] : note;
      if (!chosen) {
        target.textContent = "No candidate held " + Math.round(HOLD_CAP * 100) + "% of validation or less. The card is unchanged. Tried " + tried + ".";
        return;
      }
      target.textContent = "Fit on validation. " + modeLabel(chosen.mode) + " mark " + formatCut(chosen.distance) + ", confidence " + chosen.confidence.toFixed(2) + (chosen.keepOn ? ", keep " + chosen.keep.toFixed(3) : ", keep off") + ". Hold " + (chosen.hold * 100).toFixed(1) + "% of validation. Test did not pick this.";
    }, 30);
  }

  function distanceGrid(rows, reached, node, mode) {
    const values = [];
    rows.forEach((frame, index) => {
      if (!reached.has(index)) return;
      const scores = frame.models[node.model];
      if (!scores) return;
      if (!knownFrame(frame)) return;
      if (!callMatchesLabel(predictedClass(scores, node), frame)) return;
      const value = Number(scores[mode]);
      if (Number.isFinite(value)) values.push(value);
    });
    values.sort((left, right) => left - right);
    const distances = [];
    PORTIONS.forEach((portion) => {
      const value = quantile(values, portion);
      if (value !== null) distances.push(value);
    });
    const payload = loaded.get(node.model);
    const suggestion = payload && payload.suggestions ? Number(payload.suggestions[mode]) : NaN;
    if (Number.isFinite(suggestion)) distances.push(suggestion);
    if (!distances.length) distances.push(Number(node.distance) || 0.5);
    const unique = [];
    distances.forEach((value) => {
      const rounded = Math.round(value * 1e6) / 1e6;
      if (!unique.some((item) => Math.abs(item - rounded) < 1e-6)) unique.push(rounded);
    });
    return unique;
  }

  function quantile(sorted, portion) {
    if (!sorted.length) return null;
    const index = (sorted.length - 1) * portion;
    const low = Math.floor(index);
    const high = Math.ceil(index);
    if (low === high) return sorted[low];
    return sorted[low] * (high - index) + sorted[high] * (index - low);
  }

  function goalKey(stats, goal) {
    const balanced = stats.balanced === null ? -1 : stats.balanced;
    const accuracy = stats.accepted ? stats.correct / stats.accepted : -1;
    if (goal === "balanced") return [balanced, -stats.still, -stats.heldRight];
    if (goal === "accuracy") return [accuracy, stats.accepted, -stats.still];
    if (goal === "still") return [-stats.still, balanced, -stats.heldRight];
    return [stats.caught - stats.heldRight, balanced, -stats.holdCap];
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

  function modelNeedsBuild(rel) {
    return !modelReady(rel) || needsResinRefresh(rel);
  }

  function downstreamIds(id) {
    const seen = new Set();
    const stack = [id];
    const out = [];
    while (stack.length) {
      const current = stack.pop();
      if (!current || seen.has(current)) continue;
      seen.add(current);
      out.push(current);
      graph.wires.forEach((wire) => {
        if (wire.from === current) stack.push(wire.to);
      });
    }
    return out;
  }

  function showFromNodes(ids) {
    const next = [];
    ids.forEach((id) => {
      downstreamIds(id).forEach((item) => {
        if (next.indexOf(item) === -1) next.push(item);
      });
    });
    if (!next.length) return;
    viewNodes = next;
    listCap = LIST_STEP;
    canvas.querySelectorAll(".flow-card").forEach((card) => {
      card.classList.toggle("in-view", viewNodes.indexOf(card.dataset.node) !== -1);
    });
    paintList();
  }

  function paintScoreButton() {
    if (!scoreButton) return;
    const rels = modelRels();
    const missingModel = rels.some((rel) => modelNeedsBuild(rel));
    const beltReady = root.dataset.beltReady === "yes" || (belt && belt.ready);
    if (!beltReady) scoreButton.textContent = "Score belt checks";
    else if (missingModel) scoreButton.textContent = "Score new checkpoints";
    else scoreButton.textContent = "Rescore";
    if (scoreNote && belt && belt.ready) {
      const counts = belt.counts || {};
      scoreNote.textContent = "Belt cache: validation " + (counts.validation || 0) + ", test " + (counts.test || 0) + ", probes " + (counts.probe || 0) + ", garbage " + (counts.garbage || 0) + ".";
    }
  }

  function modelRels() {
    const rels = [];
    graph.nodes.forEach((node) => {
      if (node.type === "model" && node.model && rels.indexOf(node.model) === -1) rels.push(node.model);
    });
    return rels;
  }

  function scoreOneModel(node) {
    if (!node.model) {
      flowNote.textContent = "Pick a checkpoint first.";
      return;
    }
    const replacing = modelReady(node.model) || scoredAtLoad.has(node.model);
    const ids = graph.nodes.filter((item) => item.type === "model" && item.model === node.model).map((item) => item.id);
    onScore({ preventDefault() {} }, { rels: [node.model], rerun: replacing, skipBelt: true, focus: ids });
  }

  function hiddenField(name, value) {
    let field = scoreForm.querySelector('input[name="' + name + '"]');
    if (!value) {
      if (field) field.remove();
      return;
    }
    if (!field) {
      field = document.createElement("input");
      field.type = "hidden";
      field.name = name;
      scoreForm.appendChild(field);
    }
    field.value = value;
  }

  async function onScore(event, options) {
    event.preventDefault();
    const beltReady = root.dataset.beltReady === "yes" || (belt && belt.ready);
    const needing = modelRels().filter((rel) => modelNeedsBuild(rel));
    const rels = options && options.rels ? options.rels : (needing.length ? needing : modelRels());
    const skipBelt = Boolean(options && options.skipBelt);
    const cachesOnDisk = rels.length > 0 && rels.every((rel) => scoredAtLoad.has(rel)) && (skipBelt || beltReady);
    const replacing = options && options.rerun !== undefined ? options.rerun : cachesOnDisk;
    focusAfterScore = options && options.focus ? options.focus.slice() : graph.nodes.filter((node) => (
      node.type === "model" && rels.indexOf(node.model) !== -1
    )).map((node) => node.id);
    const host = scoreForm.querySelector("[data-score-models]");
    host.replaceChildren();
    rels.forEach((rel) => {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "model";
      input.value = rel;
      host.appendChild(input);
    });
    hiddenField("skip_belt", skipBelt ? "yes" : "");
    hiddenField("confirm", "");
    if (replacing) {
      const message = skipBelt
        ? "This scores this checkpoint again. The belt cache and the other checkpoints stay. Continue?"
        : "This scores these photos again and replaces the cache. Continue?";
      if (!window.confirm(message)) return;
      hiddenField("confirm", "yes");
    }
    if (scoreNote) scoreNote.textContent = "Starting…";
    if (scoreButton) scoreButton.disabled = true;
    let response;
    try {
      response = await fetch(scoreForm.action, {
        method: "POST",
        body: new FormData(scoreForm),
        headers: { Accept: "application/json" },
      });
    } catch (error) {
      if (scoreButton) scoreButton.disabled = false;
      if (scoreNote) scoreNote.textContent = "Could not start scoring.";
      return;
    }
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok || !payload || !payload.ok) {
      if (scoreButton) scoreButton.disabled = false;
      if (scoreNote) scoreNote.textContent = (payload && payload.error) || "Could not start scoring.";
      return;
    }
    if (scoreNote) scoreNote.textContent = "Scoring. The board fills in when the job finishes.";
    try {
      const status = await fetch("/jobs/status", { cache: "no-store" }).then((item) => item.json());
      if (status && !status.running) {
        await refreshScores();
        if (focusAfterScore.length) showFromNodes(focusAfterScore);
        focusAfterScore = [];
        if (scoreNote) {
          scoreNote.textContent = status.exit_code
            ? (status.status_line || "Scoring stopped.")
            : "Scores are in. Move a card or a cut without scoring again.";
        }
      }
    } catch (error) {
      /* the job stream still refreshes the board when it finishes */
    }
  }

  async function saveGraph(event) {
    event.preventDefault();
    const name = root.querySelector("[data-graph-name]").value.trim();
    const response = await fetch(root.dataset.saveUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, graph }),
    });
    const payload = await response.json();
    if (!payload.ok) {
      flowNote.textContent = payload.error || "Could not save that graph.";
      return;
    }
    const select = root.querySelector("[data-graph-pick]");
    if (!Array.from(select.options).some((option) => option.value === payload.name)) {
      const option = document.createElement("option");
      option.value = payload.name;
      option.textContent = payload.name;
      select.appendChild(option);
    }
    select.value = payload.name;
    localStorage.setItem(graphNameKey, payload.name);
    flowNote.textContent = "Saved " + payload.name + ".";
  }

  function applyGraph(next, name) {
    if (!next || !Array.isArray(next.nodes) || !Array.isArray(next.wires)) {
      flowNote.textContent = "That file is not a pipeline.";
      return false;
    }
    graph = next;
    bumpSerial();
    selectedId = graph.nodes[0] ? graph.nodes[0].id : "";
    pending = null;
    portFilter = null;
    viewNodes = [];
    saveDraft();
    localStorage.setItem(graphNameKey, name || "");
    const select = root.querySelector("[data-graph-pick]");
    if (select) select.value = name || "";
    paintBoard();
    paintInspector();
    run();
    return true;
  }

  async function loadGraph(name) {
    const response = await fetch(root.dataset.graphUrl.replace("NAME", encodeURIComponent(name)));
    if (!response.ok) {
      flowNote.textContent = "That graph is not on disk.";
      return;
    }
    const payload = await response.json();
    if (!payload.graph || !Array.isArray(payload.graph.nodes)) {
      flowNote.textContent = "That file is not a pipeline.";
      return;
    }
    if (!applyGraph(payload.graph, name)) return;
    await ensureModels();
    flowNote.textContent = "Loaded " + name + ".";
  }
}
