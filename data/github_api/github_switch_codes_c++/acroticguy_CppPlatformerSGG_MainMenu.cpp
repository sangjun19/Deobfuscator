#include "MainMenu.h"

MainMenu* MainMenu::instance = nullptr;

MainMenu::MainMenu() : Menu()
{
}

MainMenu* MainMenu::getInstance()
{
	if (instance == nullptr) {
		instance = new MainMenu();
	}
	return instance;
}

void MainMenu::update(float dt)
{
	float delta_t = dt / 1000.f;

	curr_cd += delta_t;

	bool cd_exceeded = (curr_cd > m_state->cooldown);

	if (graphics::getKeyState(graphics::SCANCODE_UP)) {
		if (cd_exceeded) {
			sel--;
			graphics::playSound("assets\\Menu\\gta hover.mp3", 0.2f, false);
			curr_selected = ((sel % buttons) + buttons) % buttons;
			curr_cd = 0;
		}
	}
	if (graphics::getKeyState(graphics::SCANCODE_DOWN)) {
		if (cd_exceeded) {
			sel++;
			graphics::playSound("assets\\Menu\\gta hover.mp3", 0.2f, false);
			curr_selected = ((sel % buttons) + buttons) % buttons;
			curr_cd = 0;
		}
	}
	if (graphics::getKeyState(graphics::SCANCODE_RETURN)) {
		if (cd_exceeded) {
			graphics::playSound("assets\\Menu\\clink.wav", 0.2f, false);

			//NAVIGATE TO NEXT PAGE
			curr_cd = 0;
			switch (curr_selected) {
			case 0:
				//PLAY
				m_state->in_menu = false;
				m_state->init();
				break;
			case 1:
				//CHAR SELECT
				m_state->in_char_sel = true;
				m_state->in_menu = false;
				break;
			case 2:
				m_state->in_set_diff = true;
				m_state->in_menu = false;
				break;
			case 3:
				//EXIT
				exit(0);
			}
		}
	}
}

void MainMenu::init()
{
	curr_cd = 0;
	curr_selected = ((sel % buttons) + buttons) % buttons;
	title = "MAIN MENU";

	m_brush_menu.outline_opacity = 0.f;
	m_brush_menu.texture = m_state->getAssetPath("Menu\\menu_bg.png");

	m_brush_arrow.outline_opacity = 0.f;
	m_brush_arrow.texture = m_state->getAssetPath("Menu\\arrow.png");

}

void MainMenu::draw()
{
	graphics::drawRect(m_state->getCanvasWidth() / 2, m_state->getCanvasHeight() / 2, m_state->getBackgroundWidth(), m_state->getBackgroundHeight(), m_brush_menu); // Background
	graphics::drawText(1, 1, 1, title, m_brush_menu); // Title

	for (int i = 0; i < buttons; i++) {
		graphics::drawText(1, i + 2.5, 0.5, menu_selection[i], m_brush_sel); // Draw all the buttons, with their respective brush
	}

	graphics::drawRect(0.6f, curr_selected + 2.3, -0.5, 0.25, m_brush_arrow);

	std::string control_msgs[6] = { "CONTROLS:", "WASD/ARROW KEYS TO MOVE", "ENTER/SPACE TO SHOOT", "G TO TOGGLE DEBUG MODE", "", "PRO TIP: DON'T DIE!"};

	for (int i = 0; i < 6; i++) {
		graphics::drawText(8, i/2.f + 2.5, 0.5, control_msgs[i], m_brush_sel);
	}
}
