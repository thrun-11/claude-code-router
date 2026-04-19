export interface HttpResponse {
  status: number;
  body: string;
  set_cookie_headers?: string[];
  headers?: Record<string, string>;
}

export interface StreamChunk {
  content: string;
  done: boolean;
  set_cookie_headers?: string[];
}

export type StreamCallback = (chunk: StreamChunk) => void;

export interface HttpClientOptions {
  timeoutSec?: number;
  proxyUrl?: string;
  forceHttp11?: boolean;
}

export interface HttpClient {
  get(url: string, headers: Record<string, string>, options?: HttpClientOptions): Promise<HttpResponse>;
  post(url: string, headers: Record<string, string>, body: string, options?: HttpClientOptions): Promise<HttpResponse>;
  postStream(url: string, headers: Record<string, string>, body: string, callback: StreamCallback, options?: HttpClientOptions): Promise<void>;
}