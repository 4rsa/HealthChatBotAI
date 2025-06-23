const form      = document.getElementById("chat-form");
const input     = document.getElementById("user-input");
const log       = document.getElementById("chat-log");
const fileInput = document.getElementById("tumor-image");

const API_TEXT  = "/chat";
const API_IMAGE = "/brain/segment";

/* ─── Создание пузырька ─── */
function addBubble(txt, who = "bot", html = false) {
  const d = document.createElement("div");
  d.classList.add("bubble", who);
  html ? d.innerHTML = txt : d.textContent = txt;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

/* ─── Пузырёк с анимацией загрузки (три точки) ─── */
function addLoaderBubble() {
  const d = document.createElement("div");
  d.classList.add("bubble", "bot", "loading");
  d.innerHTML = `
    <span class="dot"></span>
    <span class="dot"></span>
    <span class="dot"></span>
  `;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

/* ─── Отправка текстового сообщения ─── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;

  addBubble(msg, "user");
  input.value = "";

  const loader = addLoaderBubble();

  try {
    const res = await fetch(API_TEXT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();
    loader.remove();
    addBubble(data.answer || "No answer.", "bot");
  } catch {
    loader.remove();
    addBubble("❌ Error connecting to server.", "bot");
  }
});

/* ─── Загрузка изображения ─── */
fileInput.addEventListener("change", async () => {
  const f = fileInput.files[0];
  if (!f) return;

  const previewURL = URL.createObjectURL(f);
  const html = `🧠 Uploading:<br>
    <img src="${previewURL}" style="max-width:50%; margin-top:6px; border-radius:8px;" />`;
  addBubble(html, "user", true);

  const loader = addLoaderBubble();

  try {
    const fd = new FormData();
    fd.append("image", f);

    const res = await fetch("/brain/segment", { method: "POST", body: fd });
    const data = await res.json();
    loader.remove();

    if (data.error) {
      addBubble("❌ " + data.error, "bot");
    } else {
      const responseHTML = `
        ✅ <b>${data.prediction}</b><br>
        <img src="${data.mask_image_url}"
             style="max-width:50%; margin-top:6px; border-radius:8px;" />
      `;
      addBubble(responseHTML, "bot", true);
    }
  } catch {
    loader.remove();
    addBubble("❌ Upload failed.", "bot");
  }
});

const clearBtn = document.getElementById("clear-chat");
clearBtn.addEventListener("click", () => {
  log.innerHTML = "";
});
