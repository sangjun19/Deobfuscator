// Repository: MandavkarPranjal/lit
// File: src/tui_interface.rs

use crate::config::load_config;
use crate::input::{handle_input, InputMode};
use crossterm::{
    cursor,
    event::{self, DisableMouseCapture, EnableMouseCapture, KeyCode},
    execute,
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{self};
use std::time::{Duration, Instant};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Terminal,
};

pub fn run_tui() -> io::Result<()> {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;
    terminal::enable_raw_mode()?;

    let backend = CrosstermBackend::new(stdout.lock());
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal);

    terminal::disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app<B: tui::backend::Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut state = ListState::default();
    let mut delete_state = ListState::default();
    let mut switch_state = ListState::default();
    let mut update_state = ListState::default(); // New state for updating profiles
    let options: Vec<String> = vec![
        "Add Profile".to_string(),
        "Switch Profile".to_string(),
        "Update Profile".to_string(),
        "Delete Profile".to_string(),
        "List Profiles".to_string(),
    ];

    state.select(Some(0)); // Initialize with the first option selected
    let mut last_event_time = Instant::now();
    let mut input_mode = InputMode::Normal;
    let mut profile_name = String::new();
    let mut user_name = String::new();
    let mut user_email = String::new();
    let mut selected_profile_to_delete: Option<String> = None;
    let mut selected_profile_to_switch: Option<String> = None;
    let mut selected_profile_to_update: Option<String> = None; // New selected profile to update
    let mut delete_options: Vec<String> = Vec::new();
    let mut switch_options: Vec<String> = Vec::new();
    let mut update_options: Vec<String> = Vec::new(); // New options for updating
    let mut config = load_config();

    loop {
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([Constraint::Percentage(20), Constraint::Percentage(80)].as_ref())
                .split(size);

            match input_mode {
                InputMode::InputProfileName => {
                    let paragraph = Paragraph::new(format!("Profile Name: {}", profile_name))
                        .block(
                            Block::default()
                                .title("Enter Profile Name")
                                .borders(Borders::ALL),
                        );
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::InputUserName | InputMode::UpdateProfileUserName => {
                    let title = if input_mode == InputMode::InputUserName {
                        "Enter User Name"
                    } else {
                        "Update User Name"
                    };
                    let paragraph = Paragraph::new(format!("User Name: {}", user_name)).block(
                        Block::default()
                            .title(title)
                            .borders(Borders::ALL),
                    );
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::InputUserEmail | InputMode::UpdateProfileUserEmail => {
                    let title = if input_mode == InputMode::InputUserEmail {
                        "Enter User Email"
                    } else {
                        "Update User Email"
                    };
                    let paragraph = Paragraph::new(format!("User Email: {}", user_email)).block(
                        Block::default()
                            .title(title)
                            .borders(Borders::ALL),
                    );
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::ListingProfiles => {
                    let profile_items: Vec<ListItem> = config
                        .profiles
                        .keys()
                        .map(|p| ListItem::new(p.clone()))
                        .collect();
                    let profiles_list = List::new(profile_items)
                        .block(Block::default().title("Profiles").borders(Borders::ALL));

                    f.render_widget(profiles_list, chunks[0]);

                    let instructions = Paragraph::new("Press 'b' to go back to the main menu.")
                        .block(Block::default().title("Instructions").borders(Borders::ALL));
                    f.render_widget(instructions, chunks[1]);
                }
                InputMode::DeleteProfile => {
                    delete_options = config.profiles.keys().cloned().collect();
                    let items: Vec<ListItem> = delete_options
                        .iter()
                        .map(|o| ListItem::new(o.to_string()))
                        .collect();
                    let list = List::new(items)
                        .block(Block::default().borders(Borders::ALL).title("Select Profile to Delete"))
                        .highlight_style(
                            tui::style::Style::default().bg(tui::style::Color::Yellow),
                        );

                    f.render_stateful_widget(list, chunks[0], &mut delete_state);

                    let paragraph = Paragraph::new("Use arrow keys or 'j', 'k' to navigate, 'Enter' to select, 'b' to go back to main menu.")
                        .block(Block::default().title("Instructions").borders(Borders::ALL));
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::SwitchProfile => {
                    switch_options = config.profiles.keys().cloned().collect();
                    let items: Vec<ListItem> = switch_options
                        .iter()
                        .map(|o| ListItem::new(o.to_string()))
                        .collect();
                    let list = List::new(items)
                        .block(Block::default().borders(Borders::ALL).title("Select Profile to Switch"))
                        .highlight_style(
                            tui::style::Style::default().bg(tui::style::Color::Yellow),
                        );

                    f.render_stateful_widget(list, chunks[0], &mut switch_state);

                    let paragraph = Paragraph::new("Use arrow keys or 'j', 'k' to navigate, 'Enter' to select, 'b' to go back to main menu.")
                        .block(Block::default().title("Instructions").borders(Borders::ALL));
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::UpdateProfile => {
                    update_options = config.profiles.keys().cloned().collect();
                    let items: Vec<ListItem> = update_options
                        .iter()
                        .map(|o| ListItem::new(o.to_string()))
                        .collect();
                    let list = List::new(items)
                        .block(Block::default().borders(Borders::ALL).title("Select Profile to Update"))
                        .highlight_style(
                            tui::style::Style::default().bg(tui::style::Color::Yellow),
                        );

                    f.render_stateful_widget(list, chunks[0], &mut update_state);

                    let paragraph = Paragraph::new("Use arrow keys or 'j', 'k' to navigate, 'Enter' to select, 'b' to go back to main menu.")
                        .block(Block::default().title("Instructions").borders(Borders::ALL));
                    f.render_widget(paragraph, chunks[1]);
                }
                InputMode::ConfirmDeleteProfile => {
                    if let Some(ref profile) = selected_profile_to_delete {
                        let paragraph = Paragraph::new(format!(
                            "Are you sure you want to delete the profile '{}'? (y/n)",
                            profile
                        ))
                        .block(Block::default().title("Confirm Delete").borders(Borders::ALL));
                        f.render_widget(paragraph, chunks[1]);
                    }
                }
                InputMode::Normal => {
                    let items: Vec<ListItem> = options
                        .iter()
                        .map(|o| ListItem::new(o.to_string()))
                        .collect();
                    let list = List::new(items)
                        .block(Block::default().borders(Borders::ALL).title("Menu"))
                        .highlight_style(
                            tui::style::Style::default().bg(tui::style::Color::Yellow),
                        );

                    f.render_stateful_widget(list, chunks[0], &mut state);

                    let paragraph = Paragraph::new(
                        "Use arrow keys or 'j', 'k' to navigate, 'Enter' to select, 'q' to exit.",
                    )
                    .block(Block::default().title("Instructions").borders(Borders::ALL));
                    f.render_widget(paragraph, chunks[1]);
                }
            }
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let event::Event::Key(key) = event::read()? {
                if last_event_time.elapsed() < Duration::from_millis(200) {
                    continue; // Skip the event if it's too soon after the last one
                }
                last_event_time = Instant::now();

                if key.code == KeyCode::Char('q') {
                    break;
                }

                let options_slice: Vec<&str> = options.iter().map(AsRef::as_ref).collect();
                let _ = handle_input(
                    key,
                    &mut input_mode,
                    &mut profile_name,
                    &mut user_name,
                    &mut user_email,
                    &mut state,
                    &options_slice,
                    &mut selected_profile_to_delete,
                    &mut delete_state,
                    &delete_options,
                    &mut selected_profile_to_switch,
                    &mut switch_state,
                    &switch_options,
                    &mut selected_profile_to_update,
                    &mut update_state, // New update state
                    &update_options,
                    &mut config,
                );
            }
        }
    }
    Ok(())
}
