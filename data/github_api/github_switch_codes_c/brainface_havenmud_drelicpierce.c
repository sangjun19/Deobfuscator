#include <lib.h>
#include <damage_types.h>
#include <vendor_types.h>
inherit LIB_ITEM;

int SetCountDown(int timeLimit);
int GetCountDown();
string GetMaster();
void DoFade();
mixed DropTest(object who);
int countDown;
string SetMaster(string master);

string Master = "";

static void create() {
  item::create();
  SetKeyName("mithril rapier");
  SetMaterial( ({ "mithril" }) );
  SetShort("a shimmering mithril rapier");
  SetId( ({ "rapier" }) );
  SetAdjectives( ({ "shimmering", "mithril" }) );
  SetClass(40);
  SetMass(25);
  SetValue(0);
  SetVendorType(VT_WEAPON);
  SetWeaponType("pierce");
  SetDamageType(PIERCE|MAGIC);
  SetLong(
    "The hilt of this rapier is made from pure mithril while main portion of it is composed "
    "of magical energies bound to a sliver of mithril that serves as the blade."
    );
  SetDamagePoints(20000);
  SetDestroyOnSell(1);
  SetPreventSteal(1);
  SetPreventDrop( (: DropTest :) );
  set_heart_beat(1);
}


int eventStrike(object target) {
  int amt;
  amt = random(20);
  if (!(random(5))) {
  switch (amt) {
    case 0..10:
      message("environment",
    "%^BOLD%^%^WHITE%^The rapier seems to fade from existence for a momment.%^RESET%^",
        environment(target));
        amt += 20;
     break;
    case 11..15:
      message("environment",
      "%^BOLD%^%^BLUE%^The rapier glows with light.%^RESET%^",
         environment(target));
        amt += 35;
      break;
    case 16..20:
      message("environment",
        "%^RED%^BOLD%^An explosion of magical energies come the rapier.%^RESET%^",
        environment(target));
        amt += 40;
      break;
    }
  return item::eventStrike(target) + amt;
  }
  return item::eventStrike(target);
 }

mixed DropTest(object who) {
      send_messages( ({ "exact" }),
         "The rapier flickers out from existence.",
         this_object(), 0, who
       );
    call_out( (: eventDestruct :), 0 );
  return 0;
}

string SetMaster(string master){
        return (Master = master);
}

int SetCountDown(int timeLimit){
        return (countDown = timeLimit);
}

int GetCountDown(){
        return countDown;
}

string GetMaster(){
        return Master;
}

void heart_beat(){
    object room;
    object who;
    who = environment(this_object());

    if (!who) {
     eventDestruct();
     return;
    }  
    room = environment(who);
    if (!room) {
     eventDestruct();
     return;
    } 
    if (!living(who)){
     eventDestruct();
     return;
    }
        countDown--;
        if(!countDown) { DoFade(); }
        if(countDown < 0) { this_object()->eventDestruct(); }
}

void DoFade(){
        object caster;
        caster = environment(this_object());
        
        environment(this_object())->eventPrint("%^BOLD%^%^RED%^The rapier "
        "shimmers from this reality.%^RESET%^", caster);
        eventDestruct();
}