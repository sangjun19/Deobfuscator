#include <assert.h>
#include <ncurses.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ROWS 6
#define COLS 7

#define PLAYER_1_CH "x"
#define PLAYER_2_CH "o"
#define ANIM_TIME 500

#define ArrayCount(a) (sizeof(a) / sizeof((a)[0]))
#define InvalidCodePath() assert("Invalid code path")

typedef struct Window
{
    int twidth;
    int theight;
    int width;
    int height;
    int x;
    int y;
} Window;

Window create_window()
{
    Window win = {};

    initscr();
    timeout(-1);
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    win.width = COLS * 2;
    win.height = (ROWS + 1) * 2;

    getmaxyx(stdscr, win.theight, win.twidth);
    win.x = win.twidth / 2 - win.width / 2;
    win.y = 4 + win.theight / 2 - win.height / 2;

    return win;
}

void delete_window(Window *window)
{
    endwin();
}

typedef enum Turn
{
    TURN_1,
    TURN_2
} Turn;

typedef enum Winner
{
    WINNER_NONE,

    WINNER_1,
    WINNER_2,
    WINNER_DRAW,
} Winner;

typedef enum CellState
{
    CELL_STATE_EMPTY,
    CELL_STATE_1,
    CELL_STATE_2,
} CellState;

typedef struct GameState
{
    CellState board[ROWS * COLS];
    int drop_pos_x;

    Turn turn;
    Winner winner;
    bool exit_flag;

    int winner_x[4];
    int winner_y[4];
    bool winner_hidden;
    CellState winner_cell_state;
    bool animating;
    struct timespec animation_start_time;
} GameState;

int get_cell_state_at_safe(CellState *board, int x, int y)
{
    if (x < 0 || x >= COLS || y < 0 || y >= ROWS)
    {
        return -1;
    }
    else
    {
        return board[y * COLS + x];
    }
}

CellState get_cell_state_at(CellState *board, int x, int y)
{
    return board[y * COLS + x];
}

void set_cell_state_at(CellState *board, int x, int y, CellState cell_state)
{
    board[y * COLS + x] = cell_state;
}

int get_valid_drop_pos_y(CellState *board, int drop_pos_x)
{
    int y;

    for (y = ROWS - 1; y >= 0; --y)
    {
        CellState cell_state = get_cell_state_at(board, drop_pos_x, y);
        if (cell_state == CELL_STATE_EMPTY)
        {
            break;
        }
    }

    return y;
}

Winner get_winner_from_turn(Turn turn)
{
    switch (turn)
    {
        case TURN_1:
        {
            return WINNER_1;
        }
        case TURN_2:
        {
            return WINNER_2;
        }
    }
}

bool check_winner_horizontal(GameState *game_state, int move_x, int move_y)
{
    CellState cell_state = get_cell_state_at(game_state->board, move_x, move_y);

    int tmp_x = move_x;
    while (get_cell_state_at_safe(game_state->board, tmp_x, move_y) == cell_state)
    {
        --tmp_x;
    }

    ++tmp_x;

    int count = 0;
    while (get_cell_state_at_safe(game_state->board, tmp_x++, move_y) == cell_state)
    {
        ++count;
    }

    bool win = count == 4;
    if (win)
    {
        for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
        {
            game_state->winner_x[i] = tmp_x - i - 2;
            game_state->winner_y[i] = move_y;
        }
    }

    return win;
}

bool check_winner_vertical(GameState *game_state, int move_x, int move_y)
{
    CellState cell_state = get_cell_state_at(game_state->board, move_x, move_y);

    int tmp_y = move_y;
    while (get_cell_state_at_safe(game_state->board, move_x, tmp_y) == cell_state)
    {
        ++tmp_y;
    }

    --tmp_y;

    int count = 0;
    while (get_cell_state_at_safe(game_state->board, move_x, tmp_y--) == cell_state)
    {
        ++count;
    }

    bool win = count == 4;
    if (win)
    {
        for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
        {
            game_state->winner_x[i] = move_x;
            game_state->winner_y[i] = tmp_y + i + 2;
        }
    }

    return win;
}

bool check_winner_diag1(GameState *game_state, int move_x, int move_y)
{
    CellState cell_state = get_cell_state_at(game_state->board, move_x, move_y);

    int tmp_x = move_x;
    int tmp_y = move_y;
    while (get_cell_state_at_safe(game_state->board, tmp_x, tmp_y) == cell_state)
    {
        --tmp_x;
        ++tmp_y;
    }

    ++tmp_x;
    --tmp_y;

    int count = 0;
    while (get_cell_state_at_safe(game_state->board, tmp_x++, tmp_y--) == cell_state)
    {
        ++count;
    }

    bool win = count == 4;
    if (win)
    {
        for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
        {
            game_state->winner_x[i] = tmp_x - i - 2;
            game_state->winner_y[i] = tmp_y + i + 2;
        }
    }

    return win;
}

bool check_winner_diag2(GameState *game_state, int move_x, int move_y)
{
    CellState cell_state = get_cell_state_at(game_state->board, move_x, move_y);

    int tmp_x = move_x;
    int tmp_y = move_y;
    while (get_cell_state_at_safe(game_state->board, tmp_x, tmp_y) == cell_state)
    {
        ++tmp_x;
        ++tmp_y;
    }

    --tmp_x;
    --tmp_y;

    int count = 0;
    while (get_cell_state_at_safe(game_state->board, tmp_x--, tmp_y--) == cell_state)
    {
        ++count;
    }

    bool win = count == 4;
    if (win)
    {
        for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
        {
            game_state->winner_x[i] = tmp_x + i + 2;
            game_state->winner_y[i] = tmp_y + i + 2;
        }
    }

    return win;
}

void check_winner(GameState *game_state, int move_x, int move_y)
{
    if (check_winner_horizontal(game_state, move_x, move_y) || check_winner_vertical(game_state, move_x, move_y) ||
        check_winner_diag1(game_state, move_x, move_y) || check_winner_diag2(game_state, move_x, move_y))
    {
        game_state->winner = get_winner_from_turn(game_state->turn);
    }
    else
    {
        bool full = true;
        for (int x = 0; x < COLS; ++x)
        {
            for (int y = 0; y < ROWS; ++y)
            {
                if (get_cell_state_at(game_state->board, x, y) == CELL_STATE_EMPTY)
                {
                    full = false;
                    break;
                }
            }
        }

        if (full)
        {
            game_state->winner = WINNER_DRAW;
        }
    }
}

CellState get_cell_state_from_turn(Turn turn)
{
    switch (turn)
    {
        case TURN_1:
        {
            return CELL_STATE_1;
        }
        case TURN_2:
        {
            return CELL_STATE_2;
        }
    }
}

Turn get_next_turn(Turn turn)
{
    switch (turn)
    {
        case TURN_1:
        {
            return TURN_2;
        }
        case TURN_2:
        {
            return TURN_1;
        }
    }
}

void animate_winner(GameState *game_state)
{
    game_state->winner_hidden = false;
    game_state->animating = true;
    game_state->winner_cell_state = get_cell_state_from_turn(game_state->turn);
    clock_gettime(CLOCK_MONOTONIC_RAW, &game_state->animation_start_time);

    timeout(ANIM_TIME / 2);
}

bool is_drop_pos_valid(GameState *game_state, int pos)
{
    return get_valid_drop_pos_y(game_state->board, pos) != -1;
}

void drop(GameState *game_state)
{
    int drop_pos_y = get_valid_drop_pos_y(game_state->board, game_state->drop_pos_x);
    if (drop_pos_y == -1)
    {
        return;
    }

    set_cell_state_at(game_state->board, game_state->drop_pos_x, drop_pos_y,
                      get_cell_state_from_turn(game_state->turn));

    check_winner(game_state, game_state->drop_pos_x, drop_pos_y);

    if (game_state->winner == WINNER_NONE)
    {
        game_state->turn = get_next_turn(game_state->turn);
    }

    if (game_state->winner != WINNER_NONE && game_state->winner != WINNER_DRAW)
    {
        animate_winner(game_state);
    }
}

void reset(GameState *game_state)
{
    game_state->turn = TURN_1;
    game_state->winner = WINNER_NONE;
    game_state->drop_pos_x = 0;
    game_state->animating = false;

    memset(game_state->winner_x, 0, sizeof(game_state->winner_x));
    memset(game_state->winner_y, 0, sizeof(game_state->winner_y));
    memset(game_state->board, CELL_STATE_EMPTY, sizeof(game_state->board));

    timeout(-1);
}

void input(GameState *game_state)
{
    bool game_over = game_state->winner != WINNER_NONE;

    int ch = getch();
    switch (ch)
    {
        case 'q':
        {
            game_state->exit_flag = true;
        }
        break;

        case 'a':
        case KEY_LEFT:
        {
            if (!game_over)
            {
                if (game_state->drop_pos_x > 0)
                {
                    --game_state->drop_pos_x;
                }
            }
        }
        break;

        case 'd':
        case KEY_RIGHT:
        {
            if (!game_over)
            {
                if (game_state->drop_pos_x < COLS - 1)
                {
                    ++game_state->drop_pos_x;
                }
            }
        }
        break;

        case 's':
        case ' ':
        case KEY_DOWN:
        {
            if (!game_over)
            {
                drop(game_state);
            }
        }
        break;

        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        {
            int pos = ch - '1';
            if (!game_over)
            {
                if (is_drop_pos_valid(game_state, pos))
                {
                    game_state->drop_pos_x = pos;
                    drop(game_state);
                }
            }
        }
        break;

        case 'r':
        {
            if (game_over)
            {
                reset(game_state);
            }
        }
        break;
    }
}

void draw_line(int y, int x, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    char buffer[256];
    buffer[ArrayCount(buffer) - 1] = 0;
    vsnprintf(buffer, ArrayCount(buffer) - 1, fmt, args);
    va_end(args);

    move(y, x);
    addstr(buffer);
}

void draw_title(Window *win, int yoff)
{
    draw_line(win->y - yoff, win->x, "Connect 4");
    draw_line(win->y - (yoff - 2), win->x, "Player 1: %s", PLAYER_1_CH);
    draw_line(win->y - (yoff - 3), win->x, "Player 2: %s", PLAYER_2_CH);
}

void draw_winner(Window *win, int yoff, Winner winner)
{
    char buf[16];
    switch (winner)
    {
        case WINNER_1:
        {
            snprintf(buf, ArrayCount(buf), "Winner: %s", PLAYER_1_CH);
        }
        break;
        case WINNER_2:
        {
            snprintf(buf, ArrayCount(buf), "Winner: %s", PLAYER_2_CH);
        }
        break;

        case WINNER_DRAW:
        {
            snprintf(buf, ArrayCount(buf), "DRAW");
        }
        break;

        case WINNER_NONE:
        {
            InvalidCodePath();
        }
        break;
    }
    move(win->y - yoff, win->x);
    addstr(buf);
}

void draw_turn(Window *win, int yoff, Turn turn)
{
    char buf[16];
    snprintf(buf, ArrayCount(buf), "Turn: %s", turn == TURN_1 ? PLAYER_1_CH : PLAYER_2_CH);
    move(win->y - yoff, win->x);
    addstr(buf);
}

const char *get_player_ch(Turn turn)
{
    switch (turn)
    {
        case TURN_1:
        {
            return PLAYER_1_CH;
        }
        case TURN_2:
        {
            return PLAYER_2_CH;
        }
    }
}

bool is_pos_winner_pos(GameState *game_state, int x, int y)
{
    for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
    {
        if (x == game_state->winner_x[i] && y == game_state->winner_y[i])
        {
            return true;
        }
    }

    return false;
}

void draw_drop_pos_hints(Window *win, int yoff)
{
    char buf[COLS * 2 + 1];
    memset(buf, ' ', sizeof(buf) - 1);
    for (int i = 0; i < COLS * 2; i += 2)
    {
        buf[i] = '1' + i / 2;
    }
    move(win->y - yoff, win->x);
    buf[ArrayCount(buf) - 1] = 0;
    addstr(buf);
}

void draw(Window *win, GameState *game_state)
{
    if (game_state->animating)
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC_RAW, &now);

        uint64_t diff = (now.tv_sec - game_state->animation_start_time.tv_sec) * 1000000 +
                        (now.tv_nsec - game_state->animation_start_time.tv_nsec) / 1000000;
        if (diff > ANIM_TIME)
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &game_state->animation_start_time);
            game_state->winner_hidden = !game_state->winner_hidden;
        }

        if (game_state->winner_hidden)
        {
            for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
            {
                set_cell_state_at(game_state->board, game_state->winner_x[i], game_state->winner_y[i],
                                  CELL_STATE_EMPTY);
            }
        }
        else
        {
            for (int i = 0; i < ArrayCount(game_state->winner_x); ++i)
            {
                set_cell_state_at(game_state->board, game_state->winner_x[i], game_state->winner_y[i],
                                  game_state->winner_cell_state);
            }
        }
    }

    draw_title(win, 9);
    if (game_state->winner == WINNER_NONE)
    {
        draw_turn(win, 4, game_state->turn);
    }
    else
    {
        draw_winner(win, 4, game_state->winner);
    }

    draw_drop_pos_hints(win, 1);

    // draw drop position
    move(win->y - 2, win->x + game_state->drop_pos_x * 2);
    addstr(get_player_ch(game_state->turn));

    // draw the board
    for (int y = 0; y < ROWS; ++y)
    {
        move(win->y + y, win->x);
        for (int x = 0; x < COLS; ++x)
        {
            CellState cell_state = get_cell_state_at(game_state->board, x, y);
            switch (cell_state)
            {
                case CELL_STATE_EMPTY:
                {
                    addstr(". ");
                }
                break;
                case CELL_STATE_1:
                {
                    char buf[3] = {};
                    snprintf(buf, ArrayCount(buf), "%s ", PLAYER_1_CH);
                    addstr(buf);
                }
                break;
                case CELL_STATE_2:
                {
                    char buf[3] = {};
                    snprintf(buf, ArrayCount(buf), "%s ", PLAYER_2_CH);
                    addstr(buf);
                }
                break;
            }
        }
    }

    // draw the bottom line
    move(win->y + ROWS, win->x);
    for (int x = 0; x < COLS * 2 - 1; ++x)
    {
        addch('=');
    }
}

int main(int argc, char **argv)
{
    Window win = create_window();

    GameState game_state = {};
    reset(&game_state);

    for (;;)
    {
        erase();
        draw(&win, &game_state);

        input(&game_state);
        if (game_state.exit_flag)
        {
            break;
        }
    }

    delete_window(&win);
    return 0;
}
