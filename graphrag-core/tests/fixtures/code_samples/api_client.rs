use std::collections::HashMap;
use std::time::Duration;

/// HTTP API client for making requests to a remote service.
///
/// Supports GET, POST, and DELETE operations with configurable timeouts.
pub struct ApiClient {
    base_url: String,
    timeout: Duration,
    headers: HashMap<String, String>,
}

/// Errors that can occur during API operations.
#[derive(Debug)]
pub enum ApiError {
    /// The request timed out.
    Timeout(Duration),
    /// The server returned an error status code.
    HttpError { status: u16, message: String },
    /// A network-level error occurred.
    NetworkError(String),
    /// The response body could not be parsed.
    ParseError(String),
    /// Authentication failed.
    Unauthorized,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Timeout(d) => write!(f, "request timed out after {:?}", d),
            ApiError::HttpError { status, message } => {
                write!(f, "HTTP {} error: {}", status, message)
            }
            ApiError::NetworkError(msg) => write!(f, "network error: {}", msg),
            ApiError::ParseError(msg) => write!(f, "parse error: {}", msg),
            ApiError::Unauthorized => write!(f, "unauthorized"),
        }
    }
}

impl std::error::Error for ApiError {}

impl ApiClient {
    /// Create a new API client with the given base URL.
    pub fn new(base_url: &str) -> Self {
        ApiClient {
            base_url: base_url.to_string(),
            timeout: Duration::from_secs(30),
            headers: HashMap::new(),
        }
    }

    /// Set the request timeout duration.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add a custom header to all requests.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Fetch data from the given URL path.
    ///
    /// Sends an HTTP GET request and returns the response body as a string.
    pub async fn fetch_data(&self, path: &str) -> Result<String, ApiError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        self.execute_request("GET", &url, None).await
    }

    /// Post data to the given URL path.
    ///
    /// Sends an HTTP POST request with the given body and returns the response.
    pub async fn post_data(&self, path: &str, body: &str) -> Result<String, ApiError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        self.execute_request("POST", &url, Some(body)).await
    }

    /// Delete a resource at the given URL path.
    ///
    /// Sends an HTTP DELETE request and returns the response.
    pub async fn delete_data(&self, path: &str) -> Result<(), ApiError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        self.execute_request("DELETE", &url, None).await?;
        Ok(())
    }

    /// Internal method to execute HTTP requests.
    async fn execute_request(
        &self,
        _method: &str,
        _url: &str,
        _body: Option<&str>,
    ) -> Result<String, ApiError> {
        // Simulated implementation for testing purposes
        Ok(String::new())
    }
}

/// Response wrapper with metadata.
pub struct ApiResponse {
    pub status: u16,
    pub body: String,
    pub headers: HashMap<String, String>,
}

impl ApiResponse {
    /// Check if the response indicates success (2xx status code).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Parse the response body as JSON into the given type.
    pub fn parse_json<T>(&self) -> Result<T, ApiError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        serde_json::from_str(&self.body).map_err(|e| ApiError::ParseError(e.to_string()))
    }
}
