/*
 * score.c
 *
 * This source herein may be modified and/or distributed by anybody who
 * so desires, with the following restrictions:
 *    1.)  No portion of this notice shall be removed.
 *    2.)  Credit shall not be taken for the creation of this source.
 *    3.)  This code is not to be traded, sold, or used for personal
 *         gain or profit.
 *
 */

#include "rogue.h"
#include "extern.h"

const char *score_file = "lrogue2.scores";

void killed_by(object *monster, int other) {
	char buf[80];

	md_ignore_signals();

	if (other != QUIT) {
		rogue.gold *= 9.0 / 10.0;
	}

	if (other) {
		switch(other) {
			case HYPOTHERMIA:
				strcpy(buf, "died of hypothermia");
				break;
			case STARVATION:
				strcpy(buf, "died of starvation");
				break;
			case POISON_DART:
				strcpy(buf, "killed by a dart");
				break;
			case QUIT:
				strcpy(buf, "quit");
				break;
		}
	} else {
		strcpy(buf, "Killed by ");
		if (is_vowel(m_names[monster->m_char - 'A'][0])) {
			strcat(buf, "an ");
		} else {
			strcat(buf, "a ");
		}
		strcat(buf, m_names[monster->m_char - 'A']);
	}
	strcat(buf, " with ");
	sprintf(buf+strlen(buf), "%ld gold", rogue.gold);
	if ((!other) && show_skull) {
		clear();
		mvaddstr(4, 32, "__---------__");
		mvaddstr(5, 30, "_~             ~_");
		mvaddstr(6, 29, "/                 \\");
		mvaddstr(7, 28, "~                   ~");
		mvaddstr(8, 27, "/                     \\");
		mvaddstr(9, 27, "|    XXXX     XXXX    |");
		mvaddstr(10, 27, "|    XXXX     XXXX    |");
		mvaddstr(11, 27, "|    XXX       XXX    |");
		mvaddstr(12, 28, "\\         @         /");
		mvaddstr(13, 29, "--\\     @@@     /--");
		mvaddstr(14, 30, "| |    @@@    | |");
		mvaddstr(15, 30, "| |           | |");
		mvaddstr(16, 30, "| vvVvvvvvvvVvv |");
		mvaddstr(17, 30, "|  ^^^^^^^^^^^  |");
		mvaddstr(18, 31, "\\_           _/");
		mvaddstr(19, 33, "~---------~");
		center(21, (nick_name[0] ? nick_name : login_name));
		center(22, buf);
	} else {
		message(buf, 0);
	}
	message("", 0);
	put_scores(monster, other);
}

void win() {
	unwield(rogue.weapon);		// disarm and relax
	unwear(rogue.armor);
	un_put_on(rogue.left_ring);
	un_put_on(rogue.right_ring);

	clear();
	mvaddstr(10, 11, "@   @  @@@   @   @      @  @  @   @@@   @   @   @");
	mvaddstr(11, 11, " @ @  @   @  @   @      @  @  @  @   @  @@  @   @");
	mvaddstr(12, 11, "  @   @   @  @   @      @  @  @  @   @  @ @ @   @");
	mvaddstr(13, 11, "  @   @   @  @   @      @  @  @  @   @  @  @@");
	mvaddstr(14, 11, "  @    @@@    @@@        @@ @@    @@@   @   @   @");
	mvaddstr(17, 11, "Congratulations,  you have  been admitted  to  the");
	mvaddstr(18, 11, "Fighters' Guild.   You return home,  sell all your");
	mvaddstr(19, 11, "treasures at great profit and retire into comfort.");
	message("", 0);
	message("", 0);
	id_all();
	sell_pack();
	put_scores((object *) 0, WIN);
}

void quit(boolean from_intrpt) {
	char buf[128];
	int i, orow, ocol;
	boolean mc;

	md_ignore_signals();

	if (from_intrpt) {

		orow = rogue.row;
		ocol = rogue.col;
		mc = msg_cleared;

		for (i = 0; i < DCOLS; i++) {
			buf[i] = mvinch(0, i);
		}
	}
	check_message();
	message("really quit?", 1);
	if (rgetchar() != 'y') {
		md_heed_signals();
		check_message();
		if (from_intrpt) {
			for (i = 0; i < DCOLS; i++) {
				mvaddch(0, i, buf[i]);
			}
			msg_cleared = mc;
			move(orow, ocol);
			refresh();
		}
		return;
	}
	if (from_intrpt) {
		clean_up(byebye_string);
	}
	check_message();
	killed_by((object *) 0, QUIT);
}

void put_scores(object *monster, int other) {
	int i, n, rank = 10, x, ne = 0, found_player = -1;
	char scores[10][82];
	char n_names[10][30];
	char buf[100];
	FILE *fp;
	long s;

	if ((fp = fopen(score_file, "a+")) == NULL) {
		message("cannot read/write/create score file", 0);
		sf_error();
	}
	rewind(fp);

	for (i = 0; i < 10; i++) {
		if (((n = fread(scores[i], sizeof(char), 80, fp)) < 80) && (n != 0)) {
			sf_error();
		} else if (n != 0) {
			if ((n = fread(n_names[i], sizeof(char), 30, fp)) < 30) {
				sf_error();
			}
		} else {
			break;
		}
		ne++;
	}

	if (!score_only) {
		for (i = 0; i < ne; i++) {
			x = 5;
			while (scores[i][x] == ' ') {
				x++;
			}
			s = lget_number(scores[i] + x);

			if (rogue.gold > s) {
				rank = i;
				break;
			}
		}
		if (ne == 0) {
			rank = 0;
		} else if ((ne < 10) && (rank == 10)) {
			rank = ne;
		}
		if (rank < 10) {
			rewind(fp);
			insert_score(scores, n_names, nick_name, rank, ne, monster, other);
			if (ne < 10) {
				ne++;
			}
		}
	}

	clear();
	mvaddstr(3, 29, "Top Ten Games Of Rogue");
	mvaddstr(8, 0, "Rank   Score   Name");

	md_ignore_signals();

	for (i = 0; i < ne; i++) {
		if (i == rank) {
			standout();
		}
		if (i == 9) {
			scores[i][0] = '1';
			scores[i][1] = '0';
		} else {
			scores[i][0] = ' ';
			scores[i][1] = i + '1';
		}
		nickize(buf, scores[i], n_names[i]);
		mvaddstr(i+10, 0, buf);
		if (rank < 10) {
			fwrite(scores[i], sizeof(char), 80, fp);
			fwrite(n_names[i], sizeof(char), 30, fp);
		}
		if (i == rank) {
			standend();
		}
	}
	refresh();
	fclose(fp);
	message("", 0);
	if (score_only) message("", 0);
	clean_up("");
}

void insert_score(char scores[][82], char n_names[][30], char *n_name, int rank, int n, object *monster, boolean other) {
	int i;
	char buf[82];

	if (n > 0) {
		for (i = n; i > rank; i--) {
			if ((i < 10) && (i > 0)) {
				strcpy(scores[i], scores[i-1]);
				strcpy(n_names[i], n_names[i-1]);
			}
		}
	}
	sprintf(buf, "%2d    %6ld   %s: ", rank+1, rogue.gold, login_name);

	if (other) {
		switch(other) {
			case HYPOTHERMIA:
				strcat(buf, "died of hypothermia");
				break;
			case STARVATION:
				strcat(buf, "died of starvation");
				break;
			case POISON_DART:
				strcat(buf, "killed by a dart");
				break;
			case QUIT:
				strcat(buf, "quit");
				break;
			case WIN:
				strcat(buf, "a total winner");
				break;
		}
	} else {
		strcat(buf, "killed by ");
		if (is_vowel(m_names[monster->m_char - 'A'][0])) {
			strcat(buf, "an ");
		} else {
			strcat(buf, "a ");
		}
		strcat(buf, m_names[monster->m_char - 'A']);
	}
	sprintf(buf+strlen(buf), " on level %d ",  max_level);
	if ((other != WIN) && has_amulet()) {
		strcat(buf, "with amulet");
	}
	for (i = strlen(buf); i < 79; i++) {
		buf[i] = ' ';
	}
	buf[79] = 0;
	strcpy(scores[rank], buf);
	strcpy(n_names[rank], n_name);
}

boolean is_vowel(char ch) {
	return  (ch == 'a') ||
		(ch == 'e') ||
		(ch == 'i') ||
		(ch == 'o') ||
		(ch == 'u');
}

void sell_pack() {
	object *obj;
	int row = 2, val;
	char buf[80];

	obj = rogue.pack.next_object;

	clear();
	mvaddstr(1, 0, "Value      Item");

	while (obj) {
		if (obj->what_is != FOOD) {
			obj->identified = 1;
			val = get_value(obj);
			rogue.gold += val;

			if (row < DROWS) {
				sprintf(buf, "%5d      ", val);
				get_desc(obj, buf+11);
				mvaddstr(row++, 0, buf);
			}
		}
		obj = obj->next_object;
	}
	refresh();
	if (rogue.gold > MAX_GOLD) {
		rogue.gold = MAX_GOLD;
	}
	message("", 0);
}

int get_value(object *obj) {
	int wc;
	int val;

	wc = obj->which_kind;

	switch(obj->what_is) {
		case WEAPON:
			val = id_weapons[wc].value;
			if ((wc == ARROW) || (wc == DAGGER) || (wc == SHURIKEN) ||
					(wc == DART)) {
				val *= obj->quantity;
			}
			val += (obj->d_enchant * 85);
			val += (obj->hit_enchant * 85);
			break;
		case ARMOR:
			val = id_armors[wc].value;
			val += (obj->d_enchant * 75);
			if (obj->is_protected) {
				val += 200;
			}
			break;
		case WAND:
			val = id_wands[wc].value * (obj->class + 1);
			break;
		case SCROLL:
			val = id_scrolls[wc].value * obj->quantity;
			break;
		case POTION:
			val = id_potions[wc].value * obj->quantity;
			break;
		case AMULET:
			val = 5000;
			break;
		case RING:
			val = id_rings[wc].value * (obj->class + 1);
			break;
	}
	if (val <= 0) {
		val = 10;
	}
	return val;
}

void id_all() {
	int i;

	for (i = 0; i < SCROLLS; i++) {
		id_scrolls[i].id_status = IDENTIFIED;
	}
	for (i = 0; i < WEAPONS; i++) {
		id_weapons[i].id_status = IDENTIFIED;
	}
	for (i = 0; i < ARMORS; i++) {
		id_armors[i].id_status = IDENTIFIED;
	}
	for (i = 0; i < WANDS; i++) {
		id_wands[i].id_status = IDENTIFIED;
	}
	for (i = 0; i < POTIONS; i++) {
		id_potions[i].id_status = IDENTIFIED;
	}
}

int name_cmp(char *s1, char *s2) {
	int i = 0;
	int r;

	while(s1[i] != ':') {
		i++;
	}
	s1[i] = 0;
	r = strcmp(s1, s2);
	s1[i] = ':';
	return r;
}

void nickize(char *buf, char *score, char *n_name) {
	int i = 15, j;

	if (!n_name[0]) {
		strcpy(buf, score);
		return;
	}
	strncpy(buf, score, 16);

	while (score[i] != ':') {
		i++;
	}

	strcpy(buf+15, n_name);
	j = strlen(buf);

	while (score[i]) {
		buf[j++] = score[i++];
	}
	buf[j] = 0;
	buf[79] = 0;
}

void center(int row, char *buf) {
	int margin;

	margin = ((DCOLS - strlen(buf)) / 2);
	mvaddstr(row, margin, buf);
}

void sf_error() {
	message("", 1);
	clean_up("sorry, score file is out of order");
}
