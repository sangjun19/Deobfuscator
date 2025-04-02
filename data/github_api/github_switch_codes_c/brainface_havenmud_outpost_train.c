/*
 * Small Virtual area for Kylin Outpost
 * in the Frostmarches
 *
 *  - Melchezidek Aug 13, 2008
 */
 
#include <lib.h>
#include <domains.h>
#include "/domains/frostmarches/areas/outpost/outpost.h"

inherit LIB_ROOM;

int CheckLeave();
static void create(int x, int y) {
  string n, e, s, w;
  string long = "";
  string place = "/outpost_train/";
  string array OkEx, NoOk;
  mapping inv = ([ ]), items = ([ ]);
  ::create();
  
  /* Set Exits */
 
  if (x == 0 && y ==0) {
   n = OP_ROOM "construction_area2";
   s = FROSTMARCHES_VIRTUAL + place + x + "," + (y -1);
   e = FROSTMARCHES_VIRTUAL + place + (x + 1) + "," + y;
  }
  if (x < 5) e = FROSTMARCHES_VIRTUAL + place + (x + 1) + "," + y;
  if (x > 0) w = FROSTMARCHES_VIRTUAL + place + (x - 1) + "," + y;
  if (y > -5) s = FROSTMARCHES_VIRTUAL + place + x + "," + (y - 1);
  if (y < 0) n = FROSTMARCHES_VIRTUAL + place + x + "," + (y + 1);
  if (n) AddExit("north", n);
  if (s) AddExit("south", s);
  if (e) AddExit("east", e);
  if (w) AddExit("west", w);

long = "These training grounds have been set up to make sure the soldiers"
       " in the outpost are in the best condition possible. Many soldiers"
       " can be found here during their offtime practicing and sparring"
       " with each other to keep themselves fresh.";
    
  NoOk = ({ "north", "south", "east", "west" });
  NoOk -= GetExits();
  if (sizeof(NoOk) == 0) { long += " The training grounds continue in all directions.";   }
  if (sizeof(NoOk) == 1) { long += " The walls around the training grounds prevent movement to the "
                         + NoOk[0] + ".";   }
  if (sizeof(GetExits()) > 1 && sizeof(NoOk) > 1) {
    long += " The training grounds continue to the " + conjunction(GetExits(), "and") + ".";
  }

  switch(random(4)) {
  case 0:
   inv[OP_NPC "dwarf_paladin"] = (random(1) + 2);
   break;
  case 1:
     inv[OP_NPC "paladin"] = (random(1) + 2);
     break;
  case 2:
   inv[OP_NPC "inquisitor"] = (random(1) + 2);
   break;
  case 3:
   inv[OP_NPC "orthodox_monk"] = (random(1) + 2);
   break;
  }

  SetShort("The Training Grounds of the Outpost");
  SetClimate("sub-arctic");
  SetDomain("Frostmarches");
  SetGoMessage("You cannot go that way.");
  SetLong(long);
  SetInventory(inv);
  SetSmell( ([
       ]) );
  SetListen( ([
   ]) );
}
