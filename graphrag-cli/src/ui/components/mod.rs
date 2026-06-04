//! Reusable UI components for the TUI

use color_eyre::eyre::Result;
use ratatui::{layout::Rect, Frame};

use crate::action::Action;

pub mod help_overlay;
pub mod info_panel;
pub mod progress;
pub mod query_input;
pub mod raw_results_viewer;
pub mod results_viewer;
pub mod status_bar;

/// Component trait for UI elements
pub trait Component {
    /// Initialize the component
    #[allow(dead_code)]
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Handle an action and optionally return a new action
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        let _ = action;
        None
    }

    /// Render the component
    fn render(&mut self, f: &mut Frame, area: Rect);
}
