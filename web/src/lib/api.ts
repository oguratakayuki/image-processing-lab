// FastAPIとの通信ヘルパー。学習用ローカル環境のみを想定しているため
// 認証等は持たず、直接localhostのAPIサーバーを叩く。

export interface ImageStep {
  name: string;
  description: string;
  // "data:image/png;base64,...." 形式。そのまま <img src> に渡せる。
  image_base64: string;
}

export interface ProcessImageResponse {
  steps: ImageStep[];
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function postImage(
  path: string,
  file: File,
  fields: Record<string, string> = {}
): Promise<ProcessImageResponse> {
  const formData = new FormData();
  formData.append("file", file);
  for (const [key, value] of Object.entries(fields)) {
    formData.append(key, value);
  }

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export function runGrayscale(file: File): Promise<ProcessImageResponse> {
  return postImage("/color/grayscale", file);
}

export function runBrightnessContrast(
  file: File,
  alpha: number,
  beta: number
): Promise<ProcessImageResponse> {
  return postImage("/color/brightness-contrast", file, {
    alpha: String(alpha),
    beta: String(beta),
  });
}
