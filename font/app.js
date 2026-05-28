const messageList = document.getElementById("messageList");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");

const API_URL = "/api/chat";

function appendMessage(role, content) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const item = document.createElement("div");
  item.className = `message-bubble ${role}`;
  item.textContent = content;

  row.appendChild(item);
  messageList.appendChild(row);
  messageList.scrollTop = messageList.scrollHeight;
  return item;
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  queryInput.disabled = loading;
  sendBtn.textContent = loading ? "发送中..." : "发送";
}

function resetChat() {
  messageList.innerHTML = "";
  appendMessage("meta", "欢迎使用智能客服，请输入您的问题开始咨询。");
}

async function sendQuery(query) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || data?.error || "请求失败");
  }
  return data;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";
  setLoading(true);
  statusText.textContent = "正在生成回复...";

  try {
    const result = await sendQuery(query);
    appendMessage("assistant", result.answer || "抱歉，当前没有返回有效回复。");
    statusText.textContent = result.is_fallback ? "已使用兜底回复" : "回复完成";
  } catch (error) {
    appendMessage("assistant", `请求出错：${error.message}`);
    statusText.textContent = "请求失败";
  } finally {
    setLoading(false);
  }
});

clearBtn.addEventListener("click", () => {
  resetChat();
  statusText.textContent = "已清空对话";
});

resetChat();
