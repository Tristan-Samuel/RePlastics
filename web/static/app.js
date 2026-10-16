const log = document.querySelector("#log");
let seen = -1;
let sawRunning = document.body.dataset.running === "yes";
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
  if (state.running) sawRunning = true;
  if (
    sawRunning &&
    !state.running &&
    document.body.dataset.watchJob === "yes"
  ) {
    window.location.reload();
  }
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

document.querySelectorAll("[data-filters] select").forEach((select) => {
  select.addEventListener("change", () => {
    const root = document.querySelector("[data-filters]");
    const klass = root.querySelector("[data-filter-class]").value;
    const resin = root.querySelector("[data-filter-resin]").value;
    document.querySelectorAll("[data-true]").forEach((el) => {
      const okClass = klass === "any" || el.dataset.true === klass;
      const okResin = resin === "any" || el.dataset.resin === resin;
      el.hidden = !(okClass && okResin);
    });
  });
});
