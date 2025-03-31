// Repository: sclolus/Turbofish
// File: rust_kernel/dependencies/terminal/src/tty.rs

use super::{Cursor, Pos, SCREEN_MONAD};
use alloc::collections::vec_deque::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use ansi_escape_code::{AnsiColor, CursorMove, CSI};
use arrayvec::ArrayString;
use core::fmt::Write;
use libc_binding::local_buffer;
use screen::{AdvancedGraphic, Drawer};

/// Description of a TTY buffer
#[derive(Debug, Clone)]
struct TerminalBuffer {
    buf: VecDeque<Vec<(u8, AnsiColor)>>,
    fixed_buf: Vec<Option<(u8, AnsiColor)>>,
    draw_start_pos: usize,
    write_pos: usize,
    nb_lines: usize,
    nb_columns: usize,
}

/// Here a TTY buffer
impl TerminalBuffer {
    fn new(nb_lines: usize, nb_columns: usize, buf_max_capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(buf_max_capacity),
            fixed_buf: vec![None; nb_lines * nb_columns],
            write_pos: 0,
            nb_lines,
            nb_columns,
            draw_start_pos: 0,
        }
    }

    /// Make some place for a new screen
    fn make_place(&mut self) {
        if self.buf.len() < self.buf.capacity() {
            self.buf
                .push_back(vec![(0, AnsiColor::BLACK); self.nb_lines * self.nb_columns]);
        } else {
            let mut first = self.buf.pop_front().unwrap();
            // fresh the vec for reuse as last elem
            for c in &mut first {
                *c = (0, AnsiColor::BLACK);
            }
            self.buf.push_back(first);
            self.draw_start_pos -= self.nb_lines * self.nb_columns;
            self.write_pos -= self.nb_lines * self.nb_columns;
        }
    }

    /// Get a specified character from the buffer
    fn get_char(&self, i: usize) -> Option<(u8, AnsiColor)> {
        self.buf
            .get(i / (self.nb_lines * self.nb_columns))
            .map(|screen| screen[i % (self.nb_lines * self.nb_columns)])
    }

    /// Write a char into the buffer
    #[inline(always)]
    fn write_char(&mut self, c: char, color: AnsiColor) {
        match self
            .buf
            .get_mut(self.write_pos / (self.nb_lines * self.nb_columns))
        {
            Some(screen) => {
                let pos = self.write_pos % (self.nb_lines * self.nb_columns);
                screen[pos] = (c as u8, color);
                self.write_pos += match c {
                    '\n' => self.nb_columns - pos % self.nb_columns,
                    _ => 1,
                };
                if self.write_pos > self.draw_start_pos + self.nb_columns * self.nb_lines {
                    self.draw_start_pos += self.nb_columns;
                }
            }
            None => {
                self.make_place();
                self.write_char(c, color)
            }
        }
    }
}

/// TTY handle many write mode
#[derive(Debug, Clone, Copy)]
pub enum WriteMode {
    /// Scroll may move printed characters
    Dynamic,
    /// The string is staticly fixed
    Fixed,
}

/// Base structure of a TTY
#[derive(Debug, Clone)]
pub struct Tty {
    /// TTY is it on foreground
    pub foreground: bool,
    /// Current TTY cursor
    pub cursor: Cursor,
    /// Current Text Color
    pub text_color: AnsiColor,
    /// current Write mode
    pub write_mode: WriteMode,
    buf: TerminalBuffer,
    scroll_offset: isize,
    background: Option<Vec<u8>>,
}

/// Handle different types of scroll
#[derive(Debug, Copy, Clone)]
pub enum Scroll {
    /// One line on top
    Up,
    /// One line on bottom
    Down,
    /// Half screen on top
    HalfScreenUp,
    /// Half screen on bottom
    HalfScreenDown,
}

/// Implementation a a unique TTY
impl Tty {
    /// Constructor
    pub fn new(
        foreground: bool,
        nb_lines: usize,
        nb_columns: usize,
        max_screen_buffer: usize,
        background: Option<Vec<u8>>,
    ) -> Self {
        Self {
            foreground,
            buf: TerminalBuffer::new(nb_lines, nb_columns, max_screen_buffer),
            scroll_offset: 0,
            cursor: Cursor {
                pos: Default::default(),
                nb_lines,
                nb_columns,
                visible: true,
            },
            text_color: AnsiColor::WHITE,
            write_mode: WriteMode::Dynamic,
            background,
        }
    }

    /// Display the tty, mut be called when new activation of tty or switch
    pub fn refresh(&mut self, print_char: bool) {
        SCREEN_MONAD
            .lock()
            .draw_graphic_buffer(|buffer: *mut u8, width: usize, height: usize, bpp: usize| {
                if let Some(background) = &self.background {
                    use core::slice;
                    let buf =
                        unsafe { slice::from_raw_parts_mut(buffer, width * height * bpp / 8) };

                    for (i, elem) in buf.iter_mut().enumerate() {
                        *elem = background[i];
                    }
                }
                Ok(())
            })
            .unwrap();
        if print_char == true {
            self.print_screen(self.scroll_offset);
        } else {
            SCREEN_MONAD.lock().refresh_screen();
        }
    }

    /// set the background buffer
    pub fn set_background_buffer(&mut self, v: Vec<u8>) {
        self.background = Some(v);
    }

    /// Internal scroll
    pub fn scroll(&mut self, scroll: Scroll) {
        use Scroll::*;
        let add_scroll = match scroll {
            Up => -(self.buf.nb_columns as isize),
            Down => self.buf.nb_columns as isize,
            HalfScreenUp => -((self.buf.nb_lines / 2 * self.buf.nb_columns) as isize),
            HalfScreenDown => (self.buf.nb_lines / 2 * self.buf.nb_columns) as isize,
        };
        self.scroll_offset =
            if (self.scroll_offset + add_scroll + self.buf.draw_start_pos as isize) < 0 {
                -(self.buf.draw_start_pos as isize)
            } else {
                self.scroll_offset + add_scroll
            };
        self.print_screen(self.scroll_offset);
    }

    /// Allow a shell for example to move cursor manually
    pub fn move_cursor(&mut self, direction: CursorMove) {
        if !self.cursor.visible || !self.foreground {
            return;
        }
        // Clear the Old cursor
        self.clear_cursor();
        SCREEN_MONAD
            .lock()
            .refresh_text_line(self.cursor.pos.line)
            .unwrap();

        // Apply new cursor direction
        match direction {
            CursorMove::Forward(q) => {
                self.buf.write_pos += q;
                for _i in 0..q {
                    self.cursor.forward();
                }
            }
            CursorMove::Backward(q) => {
                self.buf.write_pos -= q;
                for _i in 0..q {
                    self.cursor.backward();
                }
            }
            CursorMove::HorizontalAbsolute(q) => {
                self.buf.write_pos += q - self.cursor.pos.column;
                self.cursor.pos.column = q;
            }
            _ => {
                unimplemented!();
            }
        }

        // Draw the new cursor
        self.draw_cursor();
        SCREEN_MONAD
            .lock()
            .refresh_text_line(self.cursor.pos.line)
            .unwrap();
    }

    /// draw cursor for the character designed by write_pos in coordinate cursor.y and cursor.x
    fn draw_cursor(&self) {
        let c = match self.buf.get_char(self.buf.write_pos) {
            None | Some((0, _)) => (' ' as u8, self.text_color),
            Some(elem) => elem,
        };
        SCREEN_MONAD
            .lock()
            .draw_cursor(c.0 as char, self.cursor.pos, c.1)
            .unwrap();
    }

    /// draw cursor for the character designed by write_pos in coordinate cursor.y and cursor.x
    fn clear_cursor(&self) {
        let c = match self.buf.get_char(self.buf.write_pos) {
            None | Some((0, _)) => (' ' as u8, self.text_color),
            Some(elem) => elem,
        };
        SCREEN_MONAD
            .lock()
            .clear_cursor(c.0 as char, self.cursor.pos, c.1)
            .unwrap();
    }

    /// re-print the entire screen
    fn print_screen(&mut self, offset: isize) {
        SCREEN_MONAD.lock().clear_screen();
        self.cursor.pos = Default::default();

        let mut pos_last_char_writen = self.buf.write_pos;
        let start_pos = if offset > 0 {
            self.buf.draw_start_pos + offset as usize
        } else {
            self.buf
                .draw_start_pos
                .checked_sub((-offset) as usize)
                .unwrap_or(0)
        };
        for j in (start_pos..start_pos + self.cursor.nb_columns * self.cursor.nb_lines)
            .step_by(self.cursor.nb_columns)
        {
            for i in j..j + self.cursor.nb_columns {
                if i >= start_pos + self.cursor.nb_columns * self.cursor.nb_lines {
                    break;
                }
                match self.buf.get_char(i) {
                    None => {
                        break;
                    }
                    Some(n) if n.0 == 0 => {
                        break;
                    }
                    Some(n) if n.0 == '\n' as u8 => {
                        self.cursor.cariage_return();
                        pos_last_char_writen =
                            i + (self.cursor.nb_columns - (i % self.cursor.nb_columns));
                        break;
                    }
                    Some(other) => {
                        SCREEN_MONAD
                            .lock()
                            .draw_character(other.0 as char, self.cursor.pos, other.1)
                            .unwrap();
                        self.cursor.forward();
                        pos_last_char_writen = i + 1;
                    }
                }
            }
        }

        if self.cursor.visible && self.buf.write_pos <= pos_last_char_writen {
            for _i in 0..pos_last_char_writen - self.buf.write_pos {
                self.cursor.backward();
            }
            self.draw_cursor();
        }

        // Display all the fixed character buffer
        for (i, elem) in self.buf.fixed_buf.iter().enumerate() {
            match *elem {
                None => {}
                Some(e) => SCREEN_MONAD
                    .lock()
                    .draw_character(
                        e.0 as char,
                        Pos {
                            line: i / self.cursor.nb_columns,
                            column: i % self.cursor.nb_columns,
                        },
                        e.1,
                    )
                    .unwrap(),
            }
        }

        SCREEN_MONAD.lock().refresh_screen();
    }

    /// Refresh line or scroll
    fn map_line(&mut self, line: usize) {
        if line == self.cursor.pos.line {
            self.scroll(Scroll::Down);
            self.scroll_offset = 0;
        } else {
            SCREEN_MONAD.lock().refresh_text_line(line).unwrap();
        }
    }
}

/// TTY implement some methods of writing
/// Dynamic: Classic behavior with buffering and scroll
/// Fixed: The text is always printed on screen
impl Write for Tty {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        if self.cursor.visible {
            self.clear_cursor();
        }

        // Make the scroll coherency
        if self.foreground && self.scroll_offset != 0 {
            self.scroll_offset = 0;
            self.print_screen(self.scroll_offset);
        }

        use ansi_escape_code::*;
        use EscapedItem::*;
        for e in iter_escaped(s) {
            match e {
                Escaped(e) => match e {
                    EscapedCode::Color(color) => self.text_color = color.into(),
                    EscapedCode::CursorMove(cursor_move) => self.move_cursor(cursor_move),
                },
                Str(s) => {
                    for c in s.chars() {
                        match self.write_mode {
                            WriteMode::Dynamic => {
                                // Ignore bullshit characters. Transmute into spaces
                                // TTY may use 0x0 value to delimitate the screen
                                if c == '\0' {
                                    continue;
                                } else {
                                    self.buf.write_char(c, self.text_color);
                                }
                            }
                            WriteMode::Fixed => {
                                self.buf.fixed_buf[self.cursor.pos.line * self.cursor.nb_columns
                                    + self.cursor.pos.column] = Some((c as u8, self.text_color))
                            }
                        };

                        if self.foreground {
                            if c != '\n' {
                                SCREEN_MONAD
                                    .lock()
                                    .draw_character(c, self.cursor.pos, self.text_color)
                                    .unwrap();
                                self.cursor.forward().map(|line| self.map_line(line));
                            } else {
                                // clear the old pos of cursor since
                                // cariage return call will forget the
                                // position of current cursor
                                if self.cursor.visible {
                                    self.clear_cursor();
                                }
                                self.cursor.cariage_return().map(|line| self.map_line(line));
                            }
                        }
                    }
                }
            }
        }
        if self.foreground {
            if self.cursor.visible {
                self.draw_cursor();
            }
            SCREEN_MONAD
                .lock()
                .refresh_text_line(self.cursor.pos.line)
                .unwrap();
        }
        Ok(())
    }
}

/// Handle a TTY with a write buffer
#[derive(Debug, Clone)]
pub struct BufferedTty {
    /// TTY contained
    pub tty: Tty,
    /// contains unfinished escaped sequence, capacity max = 256
    escaped_buf: ArrayString<[u8; Self::ESCAPED_BUF_CAPACITY]>,
    /// global ipc uid associated to the tty
    pub uid_file_op: Option<usize>,
}

impl AsRef<Tty> for BufferedTty {
    fn as_ref(&self) -> &Tty {
        &self.tty
    }
}

impl AsMut<Tty> for BufferedTty {
    fn as_mut(&mut self) -> &mut Tty {
        &mut self.tty
    }
}

impl BufferedTty {
    const ESCAPED_BUF_CAPACITY: usize = 32;

    /// Create a new buffered TTY
    pub fn new(tty: Tty) -> Self {
        Self {
            tty,
            escaped_buf: ArrayString::new(),
            uid_file_op: None,
        }
    }

    /// This method buffers an unfinished escape sequence and print appropriately.
    /// It appends `part`, a fragment of the current unfinished sequence inside escape_buf.
    /// If `complete` is true, after appending, the sequence is considered finished and flushed into the tty.
    /// Upon error appending the `part`, the current escaped_buf and the part are flushed into the tty.
    /// The buffer is then truncated to zero.
    #[inline(always)]
    fn sequence_buffering(&mut self, part: &str, complete: bool) -> core::fmt::Result {
        let mut err = self.escaped_buf.try_push_str(part).is_err();

        let ret: core::fmt::Result = if err {
            self.tty
                .write_str(&self.escaped_buf)
                .and_then(|_| self.tty.write_str(part))
        } else if complete {
            self.tty.write_str(&self.escaped_buf)
        } else {
            Ok(())
        };

        err |= ret.is_err();
        if err || complete {
            self.escaped_buf.truncate(0);
        }
        ret
    }

    pub fn refresh_screen(&mut self, buf: &[u8]) {
        let graphic_infos_result = SCREEN_MONAD.lock().query_graphic_infos();
        if let Ok((height, width, bpp)) = graphic_infos_result {
            let screen_size = height * width * bpp / 8;
            if screen_size == buf.len() {
                if let Some(v) = &mut self.tty.background {
                    v.as_mut_slice().copy_from_slice(buf);
                    if self.tty.foreground == true {
                        self.tty.refresh(false);
                    }
                } else {
                    log::warn!("No graphic vector allocated");
                }
            } else {
                log::warn!("Uncompatible len");
            }
        } else {
            log::warn!("Cannot query graphics info");
        }
    }

    pub fn get_frame_buffer_ptr<F>(&mut self, alloc_user_mem: F) -> Result<local_buffer, ()>
    where
        F: Fn(usize) -> Result<*mut u8, ()>,
    {
        let screen_monad = SCREEN_MONAD.lock();
        if let Ok((height, width, bpp)) = screen_monad.query_graphic_infos() {
            let len = height * width * bpp / 8;
            let buf = alloc_user_mem(len)?;
            Ok(local_buffer { buf, len, bpp })
        } else {
            Err(())
        }
    }
}

impl Write for BufferedTty {
    /// Fill its escaped buf when s has an unfinished escaped sequence
    /// to assure to call write_str on tty with complete escaped sequence.
    fn write_str(&mut self, mut s: &str) -> core::fmt::Result {
        let is_ascii_alphabetic = |c: char| c.is_ascii_alphabetic();

        if self.escaped_buf.len() != 0 {
            match s.find(is_ascii_alphabetic) {
                Some(index) => {
                    let (end_escaped, end_s) = s.split_at(index + 1);
                    self.sequence_buffering(end_escaped, true)?;
                    s = end_s;
                }
                None => {
                    self.sequence_buffering(s, false)?;
                    return Ok(());
                }
            }
        }

        // we look for the first char of `CSI` (the escape sequence)
        if let Some(begin_esc) = s.rfind(&CSI[0..1]) {
            if let None = s[begin_esc..].find(is_ascii_alphabetic) {
                self.sequence_buffering(&s[begin_esc..], false)?;
                s = &s[..begin_esc];
            }
        }
        self.tty.write_str(s)
    }
}
