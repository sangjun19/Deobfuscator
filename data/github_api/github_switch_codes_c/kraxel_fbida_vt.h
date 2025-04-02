extern int console_visible;

int console_switch_init(void (*suspend)(void),
                        void (*resume)(void));
void console_switch_cleanup(void);
int check_console_switch(void);

int console_activate_current(void);
