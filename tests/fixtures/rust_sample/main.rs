/// Entry point for the sample application.
fn main() {
    let config = Config::default();
    let processor = DataProcessor::new(config);
    let result = processor.process("hello world");
    println!("Result: {}", result);
}

/// Application configuration.
#[derive(Debug, Default)]
struct Config {
    pub max_items: usize,
    pub verbose: bool,
}

/// Processes data according to configuration.
struct DataProcessor {
    config: Config,
}

impl DataProcessor {
    /// Create a new processor with the given config.
    fn new(config: Config) -> Self {
        Self { config }
    }

    /// Process a string input and return the result.
    fn process(&self, input: &str) -> String {
        if self.config.verbose {
            eprintln!("Processing: {}", input);
        }
        let words: Vec<&str> = input.split_whitespace().collect();
        let truncated: Vec<&str> = words
            .into_iter()
            .take(self.config.max_items.max(1))
            .collect();
        truncated.join(" ")
    }
}

/// Helper utilities.
mod utils {
    /// Normalize a string by lowercasing and trimming whitespace.
    pub fn normalize(s: &str) -> String {
        s.trim().to_lowercase()
    }

    /// Check if a string contains only ASCII alphanumeric characters.
    pub fn is_alphanumeric(s: &str) -> bool {
        s.chars().all(|c| c.is_ascii_alphanumeric())
    }
}
