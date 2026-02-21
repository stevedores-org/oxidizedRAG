/// A simple key-value store.
pub struct KVStore {
    data: std::collections::HashMap<String, String>,
}

impl KVStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            data: std::collections::HashMap::new(),
        }
    }

    /// Insert a key-value pair. Returns the old value if key existed.
    pub fn insert(&mut self, key: String, value: String) -> Option<String> {
        self.data.insert(key, value)
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }

    /// Remove a key and return its value.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
