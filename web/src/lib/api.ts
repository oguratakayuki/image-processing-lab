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

export interface HistogramResponse {
  grayscale_image_base64: string;
  histogram: number[];
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function postImage<T>(
  path: string,
  file: File,
  fields: Record<string, string> = {}
): Promise<T> {
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
  return postImage<ProcessImageResponse>("/color/grayscale", file);
}

export function runBrightnessContrast(
  file: File,
  alpha: number,
  beta: number
): Promise<ProcessImageResponse> {
  return postImage<ProcessImageResponse>("/color/brightness-contrast", file, {
    alpha: String(alpha),
    beta: String(beta),
  });
}

export function runHistogram(file: File): Promise<HistogramResponse> {
  return postImage<HistogramResponse>("/histogram", file);
}

export function runThreshold(
  file: File,
  t: number
): Promise<ProcessImageResponse> {
  return postImage<ProcessImageResponse>("/color/threshold", file, {
    t: String(t),
  });
}

export function runConvolution(
  file: File,
  kernel: number[][]
): Promise<ProcessImageResponse> {
  return postImage<ProcessImageResponse>("/convolution/apply", file, {
    kernel: JSON.stringify(kernel),
  });
}
