/// A simple calculator with memory storage.
///
/// Supports basic arithmetic operations and memory recall.
pub struct Calculator {
    memory: f64,
}

impl Calculator {
    /// Create a new calculator with memory initialized to zero.
    pub fn new() -> Self {
        Calculator { memory: 0.0 }
    }

    /// Add a value to memory and return the new total.
    pub fn add(&mut self, x: f64) -> f64 {
        self.memory += x;
        self.memory
    }

    /// Subtract a value from memory and return the new total.
    pub fn subtract(&mut self, x: f64) -> f64 {
        self.memory -= x;
        self.memory
    }

    /// Multiply memory by a value and return the new total.
    pub fn multiply(&mut self, x: f64) -> f64 {
        self.memory *= x;
        self.memory
    }

    /// Divide memory by a value, returning an error on division by zero.
    pub fn divide(&mut self, x: f64) -> Result<f64, CalculatorError> {
        if x == 0.0 {
            return Err(CalculatorError::DivisionByZero);
        }
        self.memory /= x;
        Ok(self.memory)
    }

    /// Recall the current value stored in memory.
    pub fn recall(&self) -> f64 {
        self.memory
    }

    /// Reset memory to zero.
    pub fn clear(&mut self) {
        self.memory = 0.0;
    }
}

impl Default for Calculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during calculator operations.
#[derive(Debug, PartialEq)]
pub enum CalculatorError {
    /// Attempted to divide by zero.
    DivisionByZero,
    /// Value overflowed representable range.
    Overflow,
}

impl std::fmt::Display for CalculatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalculatorError::DivisionByZero => write!(f, "division by zero"),
            CalculatorError::Overflow => write!(f, "overflow"),
        }
    }
}

impl std::error::Error for CalculatorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mut calc = Calculator::new();
        assert_eq!(calc.add(5.0), 5.0);
        assert_eq!(calc.add(3.0), 8.0);
    }

    #[test]
    fn test_multiply() {
        let mut calc = Calculator::new();
        calc.add(4.0);
        assert_eq!(calc.multiply(3.0), 12.0);
    }

    #[test]
    fn test_divide_by_zero() {
        let mut calc = Calculator::new();
        calc.add(10.0);
        assert_eq!(calc.divide(0.0), Err(CalculatorError::DivisionByZero));
    }
}
