use napi::bindgen_prelude::*;
use napi_derive::napi;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .expect("Failed to create tokio runtime")
    })
}

type ClientKey = (Option<String>, bool);

#[napi(object)]
pub struct NativeHttpResponse {
    pub status: u16,
    pub body: String,
    pub set_cookie_headers: Vec<String>,
    pub headers: HashMap<String, String>,
}

#[napi(object)]
pub struct NativeStreamChunk {
    pub content: String,
    pub done: bool,
    pub set_cookie_headers: Vec<String>,
}

fn get_client(proxy_url: Option<&str>, force_http11: bool) -> Result<reqwest::Client> {
    static CLIENTS: OnceLock<Mutex<HashMap<ClientKey, reqwest::Client>>> = OnceLock::new();
    let cache = CLIENTS.get_or_init(|| Mutex::new(HashMap::new()));
    let key: ClientKey = (proxy_url.map(String::from), force_http11);

    let guard = cache
        .lock()
        .map_err(|e| Error::from_reason(format!("Client cache lock poisoned: {e}")))?;
    if let Some(client) = guard.get(&key) {
        return Ok(client.clone());
    }
    drop(guard);

    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(4)
        .tcp_keepalive(Duration::from_secs(30));

    if force_http11 {
        builder = builder.http1_only();
    }

    if let Some(url) = proxy_url {
        if !url.is_empty() {
            let proxy = reqwest::Proxy::all(url)
                .map_err(|e| Error::from_reason(format!("Invalid proxy URL: {e}")))?;
            builder = builder.proxy(proxy);
        }
    }

    let client = builder
        .build()
        .map_err(|e| Error::from_reason(format!("Failed to build HTTP client: {e}")))?;

    let mut guard = cache
        .lock()
        .map_err(|e| Error::from_reason(format!("Client cache lock poisoned: {e}")))?;
    Ok(guard.entry(key).or_insert(client).clone())
}

fn to_header_map(headers: &HashMap<String, String>) -> Result<HeaderMap> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (k, v) in headers {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| Error::from_reason(format!("Invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v)
            .map_err(|e| Error::from_reason(format!("Invalid header value for '{k}': {e}")))?;
        map.append(name, value);
    }
    Ok(map)
}

fn extract_set_cookie(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all("set-cookie")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .map(String::from)
        .collect()
}

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (name, value) in headers {
        let key = name.as_str().to_string();
        if let Ok(v) = value.to_str() {
            map.entry(key)
                .and_modify(|existing: &mut String| {
                    existing.push_str(", ");
                    existing.push_str(v);
                })
                .or_insert_with(|| v.to_string());
        }
    }
    map
}

#[napi]
pub fn http_get(
    url: String,
    headers: HashMap<String, String>,
    timeout_sec: Option<u32>,
    proxy_url: Option<String>,
    force_http11: Option<bool>,
) -> Result<NativeHttpResponse> {
    runtime().block_on(async {
        let client = get_client(proxy_url.as_deref(), force_http11.unwrap_or(false))?;
        let mut request = client.get(&url);

        if let Some(timeout) = timeout_sec {
            request = request.timeout(Duration::from_secs(timeout as u64));
        }

        let header_map = to_header_map(&headers)?;
        request = request.headers(header_map);

        let response = request
            .send()
            .await
            .map_err(|e| Error::from_reason(format!("HTTP request failed: {e}")))?;

        let status = response.status().as_u16();
        let set_cookie_headers = extract_set_cookie(response.headers());
        let headers = headers_to_map(response.headers());
        let body = response
            .text()
            .await
            .map_err(|e| Error::from_reason(format!("Failed to read response: {e}")))?;

        Ok(NativeHttpResponse {
            status,
            body,
            set_cookie_headers,
            headers,
        })
    })
}

#[napi]
pub fn http_post(
    url: String,
    headers: HashMap<String, String>,
    body: String,
    timeout_sec: Option<u32>,
    proxy_url: Option<String>,
    force_http11: Option<bool>,
) -> Result<NativeHttpResponse> {
    runtime().block_on(async {
        let client = get_client(proxy_url.as_deref(), force_http11.unwrap_or(false))?;
        let mut request = client.post(&url);

        if let Some(timeout) = timeout_sec {
            request = request.timeout(Duration::from_secs(timeout as u64));
        }

        let header_map = to_header_map(&headers)?;
        request = request.headers(header_map);
        request = request.body(body);

        let response = request
            .send()
            .await
            .map_err(|e| Error::from_reason(format!("HTTP request failed: {e}")))?;

        let status = response.status().as_u16();
        let set_cookie_headers = extract_set_cookie(response.headers());
        let response_headers = headers_to_map(response.headers());
        let response_body = response
            .text()
            .await
            .map_err(|e| Error::from_reason(format!("Failed to read response: {e}")))?;

        Ok(NativeHttpResponse {
            status,
            body: response_body,
            set_cookie_headers,
            headers: response_headers,
        })
    })
}

#[napi]
pub fn clear_client_cache() -> Result<()> {
    static CLIENTS: OnceLock<Mutex<HashMap<ClientKey, reqwest::Client>>> = OnceLock::new();
    if let Some(cache) = CLIENTS.get() {
        let mut guard = cache
            .lock()
            .map_err(|e| Error::from_reason(format!("Lock poisoned: {e}")))?;
        guard.clear();
    }
    Ok(())
}

#[napi]
pub fn http_post_stream(
    url: String,
    headers: HashMap<String, String>,
    body: String,
    callback: ThreadsafeFunction<NativeStreamChunk>,
    timeout_sec: Option<u32>,
    proxy_url: Option<String>,
    force_http11: Option<bool>,
) -> Result<()> {
    runtime().block_on(async {
        let client = get_client(proxy_url.as_deref(), force_http11.unwrap_or(false))?;
        let mut request = client.post(&url);

        if let Some(timeout) = timeout_sec {
            request = request.timeout(Duration::from_secs(timeout as u64));
        }

        let header_map = to_header_map(&headers)?;
        request = request.headers(header_map);
        request = request.body(body);

        let response = request
            .send()
            .await
            .map_err(|e| Error::from_reason(format!("HTTP request failed: {e}")))?;

        let set_cookie_headers = extract_set_cookie(response.headers());

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;

        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    let content = String::from_utf8_lossy(&bytes).to_string();
                    let status = callback.call(
                        Ok(NativeStreamChunk {
                            content,
                            done: false,
                            set_cookie_headers: Vec::new(),
                        }),
                        ThreadsafeFunctionCallMode::Blocking,
                    );

                    if status != napi::Status::Ok {
                        return Err(Error::from_reason(format!("Threadsafe callback failed: {status:?}")));
                    }
                }
                Err(e) => {
                    return Err(Error::from_reason(format!("Stream read failed: {e}")));
                }
            }
        }

        let status = callback.call(
            Ok(NativeStreamChunk {
                content: String::new(),
                done: true,
                set_cookie_headers,
            }),
            ThreadsafeFunctionCallMode::Blocking,
        );

        if status != napi::Status::Ok {
            return Err(Error::from_reason(format!("Threadsafe callback failed: {status:?}")));
        }

        Ok(())
    })
}
