const BASE_URL = 'http://localhost:8000';

function apiUrl(path) {
  return `${BASE_URL}${path}`;
}

function buildPostOptions(body, isFormData = false) {
  const options = {
    method: 'POST',
  };

  if (isFormData) {
    options.body = body; // FormData 直接放 body，浏览器会自动设置 Content-Type
  } else {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }

  return options;
}

async function postResponse(path, body, isFormData = false) {
  return fetch(apiUrl(path), buildPostOptions(body, isFormData));
}

// 普通请求：GET 文档列表、上传文件
export async function apiGet(path) {
  const res = await fetch(apiUrl(path));
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiPost(path, body, isFormData = false) {
  const res = await postResponse(path, body, isFormData);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// SSE 流式请求：POST /api/query，每收到一个 delta 调用 onDelta
export async function apiQueryStream({ question, top_k = 5, stream = true, conversation_history }, onDelta, onDone, onError) {
  try {
    const res = await postResponse('/api/query', { question, top_k, stream, conversation_history });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    // 关键：读取 SSE 流并按事件边界解析
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // stream: true 告诉 decoder 后续还有数据，避免多字节字符被截断
      buffer += decoder.decode(value, { stream: true });

      // 按 SSE 事件边界分割（空行分隔事件），保留最后不完整的部分
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || '';

      for (const part of parts) {
        // part 可能包含多行 data: ...，合并所有 data 行
        const lines = part.split(/\r?\n/);
        let dataLines = [];
        for (const line of lines) {
          if (line.startsWith('data:')) {
            dataLines.push(line.replace(/^data:\s?/, ''));
          }
        }
        if (dataLines.length === 0) continue;

        const dataStr = dataLines.join('\n');
        if (dataStr === '[DONE]') {
          onDone();
          return;
        }

        try {
          const data = JSON.parse(dataStr);
          onDelta(data);
        } catch (err) {
          // 忽略无法解析的非 JSON 数据
        }
      }
    }

    // flush decoder 剩余字节并处理最后残余缓冲
    buffer += decoder.decode();
    if (buffer) {
      const parts = buffer.split(/\r?\n\r?\n/);
      for (const part of parts) {
        const lines = part.split(/\r?\n/);
        let dataLines = [];
        for (const line of lines) {
          if (line.startsWith('data:')) {
            dataLines.push(line.replace(/^data:\s?/, ''));
          }
        }
        if (dataLines.length === 0) continue;

        const dataStr = dataLines.join('\n');
        if (dataStr === '[DONE]') {
          onDone();
          return;
        }

        try {
          const data = JSON.parse(dataStr);
          onDelta(data);
        } catch {
          // 忽略
        }
      }
    }
    onDone();
  } catch (e) {
    onError(e);
  }
}
