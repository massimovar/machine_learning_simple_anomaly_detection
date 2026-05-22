(function () {
  "use strict";

  var translations = {
    en: {
      documentTitle: "OptixEdge: Anomaly Detection on the Edge",
      deckAriaLabel: "OptixEdge presentation",
      railAriaLabel: "Presentation sections",
      progressAriaLabel: "Progress",
      languageSwitchAriaLabel: "Change language",
      languageButtonAriaLabels: {
        en: "Switch to English",
        it: "Switch to Italian"
      },
      headerTitle: "OptixEdge: Anomaly Detection on the Edge",
      headerSub: "FTOptix + MQTT + TFLite autoencoder",
      pointsTitle: "Key Points",
      factsTitle: "Project Facts",
      prevLabel: "Previous",
      prevAriaLabel: "Previous section",
      nextLabel: "Next",
      nextAriaLabel: "Next section",
      footerNote: "Keyboard left/right to navigate",
      steps: [
        {
          label: "Overview",
          title: "What This Project Is",
          summary: "A digital twin and an edge anomaly detector monitor a paper-clip machine and flag abnormal sensor behavior in near real time.",
          points: [
            "FTOptix publishes 10 sensor tags at 1 Hz.",
            "A Python service consumes MQTT data and scores each 30-sample window.",
            "The 3-D scene shows the anomaly state immediately."
          ],
          facts: [
            { k: "Sensor cadence", v: "1 Hz publish cycle" },
            { k: "Signal count", v: "10 sensor tags" },
            { k: "Deployment", v: "Docker containers on ARM64 edge device" }
          ]
        },
        {
          label: "Goal",
          title: "Goal",
          summary: "Let field-data speak and use it to detect multivariate pattern shifts earlier than static per-tag alarm thresholds, running entirely on the edge device.",
          points: [
            "The model learns normal behavior; no predefined alarm thresholds are required.",
            "Anomalies are flagged when reconstruction error exceeds mean(MSE) + 3x std(MSE).",
            "Current status: CPU inference. NPU path is ready when libvx_delegate.so is deployed."
          ],
          facts: [
            { k: "Detection method", v: "Autoencoder reconstruction error (MSE)" },
            { k: "Threshold rule", v: "mean(MSE) + 3 × std(MSE)" },
            { k: "Runtime location", v: "On-device, inside Docker container" }
          ]
        },
        {
          label: "Stack",
          title: "End-to-End Stack",
          summary: "FTOptix publishes sensor data over MQTT. The detector buffers, scores, and publishes alerts. The digital twin polls the result.",
          points: [
            "FTOptix publishes sensor JSON to ftoptix/paperclip/sensors.",
            "Mosquitto routes data to the detector, which windows, normalizes, and scores it.",
            "Results are published to anomaly/paperclip/alerts and shown in the 3-D scene."
          ],
          facts: [
            { k: "Broker", v: "eclipse-mosquitto:2" },
            { k: "Window size", v: "30 samples (configurable in config.yaml)" },
            { k: "Alert payload", v: "is_anomaly + anomaly_score + threshold" }
          ]
        },
        {
          label: "Technology",
          title: "Technology Choices",
          summary: "Why MQTT, why an autoencoder, why TFLite, why Docker.",
          points: [
            "MQTT keeps producer and detector decoupled with low protocol overhead.",
            "Autoencoder uses normal-only training; reconstruction error is the anomaly signal.",
            "TFLite + Docker gives a small runtime and repeatable edge deployment."
          ],
          facts: [
            { k: "Model architecture", v: "10 → 64 → 16 → 64 → 10" },
            { k: "Model size", v: "~3 482 parameters, ~50 KB .tflite file" },
            { k: "Runtime dependencies", v: "tflite-runtime, numpy, scikit-learn, paho-mqtt, pyyaml" }
          ]
        },
        {
          label: "Takeaways",
          title: "Takeaways",
          summary: "What matters from this demo.",
          points: [
            "No handcrafted static alarms: the ML model detects complex anomalous patterns.",
            "Let the data speak: leverage knowledge learned directly from field data.",
            "Your data stays yours: the ML model runs locally and offline; send only results to the cloud.",
            "Even on a 10-year-old machine, you can retrain the model setting the 'new normal'."
          ],
          facts: [
            { k: "Approach", v: "Unsupervised: trained on normal data only" },
            { k: "Operational model", v: "Continuous MQTT stream → window → score → alert" },
            { k: "Edge benefit", v: "Works offline, low latency, data stays local" }
          ]
        }
      ]
    },
    it: {
      documentTitle: "OptixEdge: Rilevamento anomalie sull'edge",
      deckAriaLabel: "Presentazione OptixEdge",
      railAriaLabel: "Sezioni della presentazione",
      progressAriaLabel: "Avanzamento",
      languageSwitchAriaLabel: "Cambia lingua",
      languageButtonAriaLabels: {
        en: "Passa all'inglese",
        it: "Passa all'italiano"
      },
      headerTitle: "OptixEdge: Rilevamento anomalie sull'edge",
      headerSub: "FTOptix + MQTT + autoencoder TFLite",
      pointsTitle: "Punti chiave",
      factsTitle: "Dati del progetto",
      prevLabel: "Precedente",
      prevAriaLabel: "Sezione precedente",
      nextLabel: "Successivo",
      nextAriaLabel: "Sezione successiva",
      footerNote: "Usa i tasti sinistra/destra per navigare",
      steps: [
        {
          label: "Panoramica",
          title: "Che cos'è questo progetto",
          summary: "Un digital twin e un rilevatore di anomalie sull'edge monitorano una macchina per graffette e segnalano comportamenti anomali dei sensori quasi in tempo reale.",
          points: [
            "FTOptix pubblica 10 tag sensore a 1 Hz.",
            "Un servizio Python consuma i dati MQTT e valuta ogni finestra di 30 campioni.",
            "La scena 3D mostra immediatamente lo stato di anomalia."
          ],
          facts: [
            { k: "Cadenza di pubblicazione", v: "Ciclo di pubblicazione a 1 Hz" },
            { k: "Numero di segnali", v: "10 tag sensore" },
            { k: "Distribuzione", v: "Container Docker su dispositivo edge ARM64" }
          ]
        },
        {
          label: "Obiettivo",
          title: "Obiettivo",
          summary: "Lasciare che i dati di campo parlino e usarli per rilevare variazioni di pattern multivariati prima delle soglie di allarme statiche per singolo tag, eseguendo tutto sul dispositivo edge.",
          points: [
            "Il modello apprende il comportamento normale; non sono necessarie soglie di allarme predefinite.",
            "Le anomalie vengono segnalate quando l'errore di ricostruzione supera mean(MSE) + 3x std(MSE).",
            "Stato attuale: inferenza su CPU. Il percorso NPU è pronto quando viene distribuito libvx_delegate.so."
          ],
          facts: [
            { k: "Metodo di rilevamento", v: "Errore di ricostruzione dell'autoencoder (MSE)" },
            { k: "Regola della soglia", v: "mean(MSE) + 3 × std(MSE)" },
            { k: "Esecuzione", v: "Sul dispositivo, dentro un container Docker" }
          ]
        },
        {
          label: "Architettura",
          title: "Architettura end-to-end",
          summary: "FTOptix pubblica i dati dei sensori via MQTT. Il rilevatore mette in buffer, valuta e pubblica gli allarmi. Il digital twin interroga il risultato.",
          points: [
            "FTOptix pubblica il JSON dei sensori su ftoptix/paperclip/sensors.",
            "Mosquitto instrada i dati al rilevatore, che crea le finestre, normalizza e valuta.",
            "I risultati vengono pubblicati su anomaly/paperclip/alerts e mostrati nella scena 3D."
          ],
          facts: [
            { k: "Broker", v: "eclipse-mosquitto:2" },
            { k: "Dimensione finestra", v: "30 campioni (configurabile in config.yaml)" },
            { k: "Payload allarme", v: "is_anomaly + anomaly_score + threshold" }
          ]
        },
        {
          label: "Tecnologia",
          title: "Scelte tecnologiche",
          summary: "Perché MQTT, perché un autoencoder, perché TFLite, perché Docker.",
          points: [
            "MQTT mantiene disaccoppiati produttore e rilevatore con un basso overhead di protocollo.",
            "L'autoencoder usa addestramento solo su dati normali; l'errore di ricostruzione è il segnale di anomalia.",
            "TFLite + Docker offrono un runtime leggero e un deployment edge ripetibile."
          ],
          facts: [
            { k: "Architettura del modello", v: "10 → 64 → 16 → 64 → 10" },
            { k: "Dimensione del modello", v: "~3 482 parametri, file .tflite da ~50 KB" },
            { k: "Dipendenze runtime", v: "tflite-runtime, numpy, scikit-learn, paho-mqtt, pyyaml" }
          ]
        },
        {
          label: "Conclusioni",
          title: "Messaggi chiave",
          summary: "Cosa conta in questa demo.",
          points: [
            "Nessun allarme statico costruito a mano: il modello ML rileva pattern anomali complessi.",
            "Lasciamo parlare i dati: sfruttiamo la conoscenza appresa direttamente dai dati di campo.",
            "I dati restano vostri: il modello ML gira localmente e offline; al cloud invia solo i risultati.",
            "Anche su una macchina di 10 anni fa, puoi riaddestrare il modello impostando la 'nuova normalità'."
          ],
          facts: [
            { k: "Approccio", v: "Non supervisionato: addestrato solo su dati normali" },
            { k: "Modello operativo", v: "Flusso MQTT continuo → finestra → punteggio → allarme" },
            { k: "Vantaggio edge", v: "Funziona offline, bassa latenza, i dati restano locali" }
          ]
        }
      ]
    }
  };

  var deckEl = document.getElementById("deck");
  var railEl = document.getElementById("step-rail");
  var cardEl = document.getElementById("slide-card");
  var headerTitleEl = document.getElementById("header-title");
  var headerSubEl = document.getElementById("header-sub");
  var kickerEl = document.getElementById("slide-kicker");
  var titleEl = document.getElementById("slide-title");
  var summaryEl = document.getElementById("slide-summary");
  var pointsTitleEl = document.getElementById("slide-points-title");
  var pointsEl = document.getElementById("points-list");
  var factsTitleEl = document.getElementById("slide-facts-title");
  var factsGridEl = document.getElementById("facts-grid");
  var prevBtn = document.getElementById("prev-btn");
  var nextBtn = document.getElementById("next-btn");
  var progressWrapEl = document.getElementById("progress-wrap");
  var progressEl = document.getElementById("progress-bar");
  var footerNoteEl = document.getElementById("footer-note");
  var langSwitchEl = document.getElementById("lang-switch");
  var langButtons = {
    en: document.getElementById("lang-en-btn"),
    it: document.getElementById("lang-it-btn")
  };

  var activeLanguage = "en";
  var index = 0;
  var chips = [];

  function getTranslation(language) {
    return translations[language || activeLanguage] || translations.en;
  }

  function getSteps() {
    return getTranslation().steps;
  }

  function clampIndex(newIndex) {
    var steps = getSteps();

    if (newIndex < 0) {
      return 0;
    }

    if (newIndex >= steps.length) {
      return steps.length - 1;
    }

    return newIndex;
  }

  function updateLanguageButtons() {
    Object.keys(langButtons).forEach(function (language) {
      var button = langButtons[language];
      var isActive = language === activeLanguage;

      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  function updateStaticText() {
    var translation = getTranslation();

    document.title = translation.documentTitle;
    document.documentElement.lang = activeLanguage;
    deckEl.setAttribute("aria-label", translation.deckAriaLabel);
    railEl.setAttribute("aria-label", translation.railAriaLabel);
    progressWrapEl.setAttribute("aria-label", translation.progressAriaLabel);
    langSwitchEl.setAttribute("aria-label", translation.languageSwitchAriaLabel);

    headerTitleEl.textContent = translation.headerTitle;
    headerSubEl.textContent = translation.headerSub;
    pointsTitleEl.textContent = translation.pointsTitle;
    factsTitleEl.textContent = translation.factsTitle;
    prevBtn.textContent = translation.prevLabel;
    prevBtn.setAttribute("aria-label", translation.prevAriaLabel);
    nextBtn.textContent = translation.nextLabel;
    nextBtn.setAttribute("aria-label", translation.nextAriaLabel);
    footerNoteEl.textContent = translation.footerNote;
    langButtons.en.setAttribute("aria-label", translation.languageButtonAriaLabels.en);
    langButtons.it.setAttribute("aria-label", translation.languageButtonAriaLabels.it);
  }

  function buildRail() {
    var steps = getSteps();

    railEl.innerHTML = "";
    chips = steps.map(function (step, stepIndex) {
      var button = document.createElement("button");
      var number = document.createElement("strong");

      button.type = "button";
      button.className = "step-chip";
      button.setAttribute("aria-label", (stepIndex + 1) + ". " + step.label);

      number.textContent = (stepIndex + 1) + ". ";
      button.appendChild(number);
      button.appendChild(document.createTextNode(step.label));

      button.addEventListener("click", function () {
        setStep(stepIndex);
      });

      railEl.appendChild(button);
      return button;
    });
  }

  function renderFacts(facts) {
    factsGridEl.innerHTML = "";

    facts.forEach(function (fact, factIndex) {
      var tile = document.createElement("div");
      var key = document.createElement("div");
      var value = document.createElement("div");

      tile.className = "fact-tile";
      tile.style.animationDelay = (factIndex * 70) + "ms";

      key.className = "fact-label";
      key.textContent = fact.k;

      value.className = "fact-value";
      value.textContent = fact.v;

      tile.appendChild(key);
      tile.appendChild(value);
      factsGridEl.appendChild(tile);
    });
  }

  function renderPoints(points) {
    pointsEl.innerHTML = "";

    points.forEach(function (point) {
      var item = document.createElement("li");

      item.textContent = point;
      pointsEl.appendChild(item);
    });
  }

  function shouldUseSingleColumn(step) {
    if (!step || !Array.isArray(step.points) || !Array.isArray(step.facts)) {
      return false;
    }

    if (step.facts.length < 3) {
      return true;
    }

    return step.points.some(function (point) {
      return String(point).length > 92;
    });
  }

  function setStep(newIndex) {
    var steps = getSteps();
    var step;

    index = clampIndex(newIndex);
    step = steps[index];

    cardEl.classList.remove("slide-anim");
    void cardEl.offsetWidth;
    cardEl.classList.add("slide-anim");

    kickerEl.textContent = (index + 1) + " / " + steps.length;
    titleEl.textContent = step.title;
    summaryEl.textContent = step.summary;

    renderPoints(step.points);
    renderFacts(step.facts);
    cardEl.classList.toggle("single-col", shouldUseSingleColumn(step));

    chips.forEach(function (chip, chipIndex) {
      chip.classList.toggle("active", chipIndex === index);
    });

    if (chips[index] && window.matchMedia("(max-width: 1060px)").matches) {
      chips[index].scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center"
      });
    }

    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === steps.length - 1;
    progressEl.style.width = (((index + 1) / steps.length) * 100) + "%";
  }

  function setLanguage(language) {
    if (!translations[language]) {
      return;
    }

    if (language === activeLanguage) {
      updateLanguageButtons();
      return;
    }

    activeLanguage = language;
    updateStaticText();
    buildRail();
    setStep(index);
    updateLanguageButtons();
  }

  prevBtn.addEventListener("click", function () {
    setStep(index - 1);
  });

  nextBtn.addEventListener("click", function () {
    setStep(index + 1);
  });

  window.addEventListener("keydown", function (event) {
    if (event.key === "ArrowLeft") {
      setStep(index - 1);
    }

    if (event.key === "ArrowRight") {
      setStep(index + 1);
    }
  });

  Object.keys(langButtons).forEach(function (language) {
    langButtons[language].addEventListener("click", function () {
      setLanguage(language);
    });
  });

  updateStaticText();
  buildRail();
  setStep(0);
  updateLanguageButtons();
})();
