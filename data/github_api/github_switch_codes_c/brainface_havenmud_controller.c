#include <lib.h>
#include <domains.h>
#include <std.h>
#include "../durgoroth.h"

inherit LIB_CONTROLLER;

void DoSomething();
void DoCombatSomething();

static void create() {
  ::create();
  SetKeyName("mictlan");
  SetShort("Mictlan, Keeper of Tasks"); 
  SetId( ({ "mictlan"}) );
  SetAdjectives( ({ "keeper", "tasks"}) );
  SetRace("human");
  SetBaseLanguage("Gryshnak");
  SetClass("necromancer");
  SetLevel(230);
  SetGender("male");
  SetTown("Durgoroth");
  SetReligion("Saahagoth");
  SetLong(
    "Mictlan is the only survivor from when the daemons invaded Durgoroth. "
    "It was his faith in Taigis and his magical abilities that let the daemons "
    "invade from the lower planes. Mictlan survived only by killing those "
    "very same daemons he let loose. Mictlan is now feared member of the daemon "
    "population within Durgoroth and in charge of its defenses. If you are "
    "interested in helping the daemons defend the ruins, <request quest from "
    "mictlan> or <ask mictlan for quest>." );
  SetMorality(-2000);
  SetInventory( ([
    DURG_OBJ + "bone_armour"  : "wear armour",
    DURG_OBJ + "skull_helmet" : "wear helmet",
    DURG_OBJ + "necro_knife"  : "wield knife",
    ]) );
  SetCombatAction(60, (: DoCombatSomething :));
  SetAction(1, ({
    "!speak You could ask me for a quest.",
    "!speak You could request quests from me.",
    }) );
  SetSpellBook( ([
    "poison explosion"   : 100,
    "death"              : 100,
    "temporal explosion" : 100,
    "flamestrike"        : 100,
    "curse"              : 100,
    "temperature shield" : 100,
    "aetheric rift"      : 100,
     ]) );
  SetFirstCommands( ({ 
    "cast temperature shield",
    "cast temperature shield",
    "cast temperature shield",
    "cast temperature shield",
    "cast aetheric rift",
    }) );  
}

void eventGreet(object who) {
  ::eventGreet(who);
  SetAction(15, ({
    "!speak You could ask me for a quest.",
    "!speak You could request quests from me.",
  }) );
}


void DoCombatSomething() {
  object target = GetCurrentEnemy();
  object who = this_object();
     
  if((sizeof(GetMagicProtection())) < 5) {
    eventForce("cast temperature shield");
    return;
  }

  
  switch(random(6)) {
    case 0:
      eventForce("cast curse on " + target->GetKeyName());
      break;
    case 1:
      eventForce("cast poison explosion");
      break;
    case 2:
      eventForce("cast death");
      break;
    case 3:
      eventForce("cast temporal explosion");
      break;
    default:
      eventForce("cast flamestrike");
      break;
    }
}   
void heart_beat() {
  ::heart_beat();
  if (sizeof(GetMagicProtectionNames()) < 5) {
    eventForce("cast temperature shield");

  }
}
