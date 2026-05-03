(function () {
  const video = document.getElementById("live-video");
  const overlay = document.getElementById("live-overlay");
  const capCanvas = document.createElement("canvas");
  const statusEl = document.getElementById("live-status");
  const modeRadios = document.querySelectorAll('input[name="live_mode"]');
  const devicePanel = document.getElementById("device-live");
  const serverPanel = document.getElementById("server-live");

  let stream = null;
  let timer = null;
  let busy = false;

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
  }

  function stopDevice() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (stream) {
      stream.getTracks().forEach(function (t) {
        t.stop();
      });
      stream = null;
    }
    if (video) video.srcObject = null;
    const ctx = overlay && overlay.getContext("2d");
    if (ctx && overlay) ctx.clearRect(0, 0, overlay.width, overlay.height);
    busy = false;
  }

  function drawOverlay(data) {
    if (!overlay || !video || !video.videoWidth) return;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (data.no_model) {
      setStatus("No trained model — add members first.");
      return;
    }
    setStatus("");
    (data.faces || []).forEach(function (f) {
      ctx.strokeStyle = f.unknown ? "#c85450" : "#6a9a6a";
      ctx.lineWidth = 2;
      ctx.strokeRect(f.x, f.y, f.w, f.h);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "14px Noto Sans JP, sans-serif";
      var label = f.role + " | " + f.identity + " | d=" + Math.round(f.distance);
      ctx.fillText(label, f.x, Math.max(f.y - 6, 16));
    });
  }

  async function captureAndAnalyze() {
    if (!video || busy || !video.videoWidth) return;
    busy = true;
    capCanvas.width = video.videoWidth;
    capCanvas.height = video.videoHeight;
    var capCtx = capCanvas.getContext("2d");
    capCtx.drawImage(video, 0, 0);
    capCanvas.toBlob(
      function (blob) {
        if (!blob) {
          busy = false;
          return;
        }
        var fd = new FormData();
        fd.append("file", blob, "frame.jpg");
        fetch("/api/analyze_frame", { method: "POST", body: fd })
          .then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          })
          .then(drawOverlay)
          .catch(function () {
            setStatus("Could not analyze frame (network or server error).");
          })
          .finally(function () {
            busy = false;
          });
      },
      "image/jpeg",
      0.82
    );
  }

  async function startDevice() {
    stopDevice();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera API not available in this browser.");
      return;
    }
    setStatus("Requesting camera…");
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
    } catch (e1) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (e2) {
        setStatus(
          "Camera blocked or unavailable. On phones, use HTTPS or allow permissions. Try Server PC mode."
        );
        return;
      }
    }
    video.srcObject = stream;
    video.playsInline = true;
    await video.play();
    setStatus("");
    timer = setInterval(captureAndAnalyze, 280);
  }

  function syncMode() {
    var mode = "device";
    modeRadios.forEach(function (r) {
      if (r.checked) mode = r.value;
    });
    if (mode === "device") {
      devicePanel.hidden = false;
      serverPanel.hidden = true;
      startDevice();
    } else {
      stopDevice();
      devicePanel.hidden = true;
      serverPanel.hidden = false;
      setStatus("");
    }
  }

  modeRadios.forEach(function (r) {
    r.addEventListener("change", syncMode);
  });

  window.addEventListener("beforeunload", stopDevice);

  if (document.querySelector('input[name="live_mode"]:checked')?.value === "device") {
    syncMode();
  }
})();
