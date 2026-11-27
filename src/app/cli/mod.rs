mod helpers;
mod options;
mod parse;
mod types;
mod validate;

pub use helpers::*;
pub use options::*;
pub use parse::parse_cli_options;
pub use types::*;
pub use validate::*;

// The mode → flag table is shared with the decision mode's own check, which
// lives outside this module.
pub(crate) use parse::{flag_is_allowed_for, Mode};

#[cfg(test)]
mod tests;
