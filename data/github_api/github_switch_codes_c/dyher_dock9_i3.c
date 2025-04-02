/* i3.c

   Tacitus@LPUniversity
   24-MAR-06
   I3 Channel Module

   Last edited on November 12th, 2007 by Tricky
   Actually, I gutted it and re-wrote it -- Tricky
 
*/

/* Socket stuff */
#include <socket.h>

#ifndef SOCKET_D
#define SOCKET_D "/adm/daemons/socket_d"
#endif

/* I3 defines */
#include <intermud3.h>

#include <driver/localtime.h>

/* Private Variables */
privatev mapping routerList = ([
  0: ([ "name": "*i4", "ip": "204.209.44.3", "port": 8080 ]),
  1: ([ "name": "*wpr", "ip": "195.242.99.94", "port": 8080 ]),
  2: ([ "name": "*adsr", "ip": "81.138.64.235", "port": 9000 ]),
]);

#define I3_DEF_ROUTER_NAME   (routerList[0]["name"])
#define I3_DEF_ROUTER_IP     (routerList[0]["ip"])
#define I3_DEF_ROUTER_PORT   (routerList[0]["port"])

privatev mapping  services;

/* Heartbeat timer */
privatev int      hb_timer;

/* Idle timer */
privatev int      idleTimeout;

privatev string array Days = ({
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
});

privatev string array Months = ({
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
});

privatev mapping SpecialOrdinals = ([
  0: "th",
  1: "st",
  2: "nd",
  3: "rd",
]);

/* Public Variables */
mapping          data, packetHook;

/* Public prototypes */
void sendToRouter(string i3Type, string origUser, mixed *packet);
void sendToRouterQ(string i3Type, string origUser, mixed *packet);
void sendToMud(string i3Type, string origUser, string targMud, mixed *packet);
void sendToMudQ(string i3Type, string origUser, string targMud, mixed *packet);
void sendToUser(string i3Type, string origUser, string targMud, string targUser, mixed *packet);
void sendToUserQ(string i3Type, string origUser, string targMud, string targUser, mixed *packet);
void sendToAll(string i3Type, string origUser, mixed *packet);
void sendToAllQ(string i3Type, string origUser, mixed *packet);
void initData();
void addRouter(string routerName, string routerIP, int routerPort);
void setData(mixed key, mixed val);
void installService(string service);
void installServices();
void uninstallService(string service);
void uninstallServices();
void add_service_name(string name, mixed flag);
void remove_service_name(string name);
void add_service(string type, string callback);
void remove_service(string type);
void setup_services();
void setup_other();
int setup_i3_socket();
void connect();
void reconnect();
void shutDown(int restart_delay);
void close_callback(int fd);
void read_callback(int fd, mixed packet);

void create()
{
    int x;

    if (previous_object())
        I3LOG("Notice", "I3", "create", "Reloading with previous_object being: " + file_name(previous_object()));
    else
        I3LOG("Notice", "I3", "create", "Reloading...");

    /* Initialize the data mapping and add the router we will use */
    initData();

    if(file_exists(I3_DATA_FILE + ".o")) restore_object(I3_DATA_FILE);

    if (!packetHook) packetHook = ([ ]);
    if (!data["router_list"])
        addRouter(I3_DEF_ROUTER_NAME, I3_DEF_ROUTER_IP, I3_DEF_ROUTER_PORT);

    services = ([ ]);

    if((x = setup_i3_socket()) < 0)
    {

        if(x == EEISCONN)
        {
            I3LOG("Warning", "I3", "create", "Already connected, maybe still closing. Will try again.");
            call_out("create", 5);

            return;
        }

        I3LOG("Error", "I3", "create", "Unable to setup socket - " + socket_error(x));
        destruct();

        return;
    }

    data["reconnect_wait"] = MIN_RETRY_TIME;

    if(file_exists(I3_CHANLIST_FILE + ".o")) rm(I3_CHANLIST_FILE + ".o");
    if(file_exists(I3_MUDLIST_FILE + ".o")) rm(I3_MUDLIST_FILE + ".o");

    setup_services();
    setup_other();

    call_out("connect", 2);

    idleTimeout = time();
    hb_timer = 0;
    set_heart_beat(30);

    I3LOG("Success", "I3", "create", "Intermud3 Communications Active (" + query_privs(this_object()) + ")");

    save_object(I3_DATA_FILE);
}

varargs void remove(int arg)
{
    save_object(I3_DATA_FILE);

    if(undefinedp(arg)) arg = 0;

    shutDown(arg);

    I3LOG("Warning", "I3", "remove", "Destructing.");
    destruct();

    return;
}

void heart_beat()
{
    hb_timer++;

    /* Every 15 minutes, force an auto-save */
    if(hb_timer == 14)
    {
        hb_timer = 0;
        save_object(I3_DATA_FILE);
    }

}

varargs void send_packet(mixed packet, int quiet)
{
    object sock = find_object(SOCKET_D);
    string visname;
    int index;

    if (stringp(packet[I3_O_USER]))
        packet[I3_O_USER] = lower_case(packet[I3_O_USER]);

    if (stringp(packet[I3_T_USER]))
        packet[I3_T_USER] = lower_case(packet[I3_T_USER]);

    switch (packet[I3_TYPE])
    {
        case "tell":
        case "emoteto":
            visname = packet[6];
            index = 6;
            break;
        case "channel-m":
        case "channel-e":
            visname = packet[7];
            index = 7;
            break;
        case "channel-t":
            visname = packet[11];
            index = 11;
            break;
    }

    if (!undefinedp(visname) && find_living(packet[I3_O_USER]))
    {
        object user = find_living(packet[I3_O_USER]);

        if (user->is_bot() && user->query_bot_name(packet[I3_O_USER] != visname))
        {
            quiet = 0;
            packet[index] = user->query_bot_name(packet[I3_O_USER]);
        }
        else if (user->query_cap_name() != visname)
        {
            quiet = 0;
            packet[index] = user->query_cap_name();
        }
    }

    if (!quiet)
    {
        I3LOG(
          "Notice",
          "I3",
          "send_packet",
          sprintf("Sending %O", packet)
        );
    }

    sock->client_write("I3", packet);
}

varargs void send_packets(mixed *packets, int quiet)
{
    foreach(mixed *packet in packets) send_packet(packet, quiet);
}

varargs void sendPacket(string i3Type, string origUser, string targMud, string targUser, mixed packet, int quiet)
{
    if (undefinedp(quiet)) quiet = 0;
    if (undefinedp(packet) || packet == 0) packet = ({ });
    if (!arrayp(packet)) packet = ({ packet });

    if (objectp(this_user()) && !this_user()->is_bot() && origUser != 0)
        origUser = query_privs(this_user());

    send_packet( ({
      i3Type,
      5,
      I3_MUD_NAME,
      origUser,
      targMud,
      targUser,
    }) + packet, quiet);
}

void sendToRouter(string i3Type, string origUser, mixed packet)
{
    sendPacket(i3Type, origUser, data["router_list"][0][0], 0, packet);
}

void sendToRouterQ(string i3Type, string origUser, mixed packet)
{
    sendPacket(i3Type, origUser, data["router_list"][0][0], 0, packet, 1);
}

void sendToMud(string i3Type, string origUser, string targMud, mixed packet)
{
    sendPacket(i3Type, origUser, targMud, 0, packet);
}

void sendToMudQ(string i3Type, string origUser, string targMud, mixed packet)
{
    sendPacket(i3Type, origUser, targMud, 0, packet, 1);
}

void sendToUser(string i3Type, string origUser, string targMud, string targUser, mixed packet)
{
    sendPacket(i3Type, origUser, targMud, targUser, packet);
}

void sendToUserQ(string i3Type, string origUser, string targMud, string targUser, mixed packet)
{
    sendPacket(i3Type, origUser, targMud, targUser, packet, 1);
}

void sendToAll(string i3Type, string origUser, mixed packet)
{
    sendPacket(i3Type, origUser, 0, 0, packet);
}

void sendToAllQ(string i3Type, string origUser, mixed packet)
{
    sendPacket(i3Type, origUser, 0, 0, packet, 1);
}

void initData()
{
    data = ([
      "router_password": 0,
      "mudlist_id": 0,
      "chanlist_id": 0,
      "services": ([ ]),
      "other": ([ ]),
    ]);
}

void addRouter(string routerName, string routerIP, int routerPort)
{
    mixed *tmp;

    if(!data["router_list"]) data["router_list"] = ({ });

    tmp = data["router_list"] - ({ ({ routerName, routerIP + " " + routerPort }) });
    data["router_list"] = ({ ({ routerName, routerIP + " " + routerPort }) }) + tmp;
}

int getIdleTimeout() { return idleTimeout; }

mapping getAllData() { return copy(data); }

mixed getData(string key) { return copy(data[key]); }

void setAllData(mapping d)
{
    data = copy(d);
    save_object(I3_DATA_FILE);
}

void setData(mixed key, mixed val)
{
    data[key] = copy(val);
    save_object(I3_DATA_FILE);
}

void addData(mixed key, mixed val)
{
    data[key] += copy(val);
    save_object(I3_DATA_FILE);
}

/* For compatibility reasons */
mapping get_mudlist() { return find_object(I3_MUDLIST)->get_mud_list(); }

void add_packetHook(string packetType, string callback)
{
    if (!packetHook[packetType]) packetHook[packetType] = ([ ]);

    packetHook[packetType] += ([
      file_name(previous_object()): callback,
    ]);

    save_object(I3_DATA_FILE);
}

int remove_packetHook(string packetType, mixed obj)
{
    if (!packetHook[packetType]) return 0;
    if (undefinedp(obj) || obj == 0) obj = previous_object();
    if (objectp(obj)) obj = file_name(obj);
    if (!packetHook[packetType][obj]) return 0;

    map_delete(packetHook[packetType], obj);
    save_object(I3_DATA_FILE);

    return 1;
}

void clean_packetHooks()
{
    foreach (mixed k, mixed v in packetHook)
        foreach (mixed kv, mixed vv in v)
            if (!find_object(kv))
                remove_packetHook(k, kv);
}

void installService(string service_fname)
{
    mixed file_stat;
    string fname = __DIR__ + "i3services/" + service_fname;

    file_stat = stat(fname + ".c");

    if (arrayp(file_stat) && sizeof(file_stat))
    {
        if(file_stat[2] != 0)
            uninstallService(service_fname);

        ANNOUNCE_CHDMOD->rec_m_msg(
          "intermud3",
          "I3",
          "Loading service '" + service_fname + "'"
        );
        I3LOG("Notice", "I3", "installService", "Loading service '" + service_fname + "'");
        catch(load_object(fname));
    }
}

void installServices()
{
    foreach (string fname in get_dir(__DIR__ + "i3services/mod_*.c"))
        installService(fname[0..<3]);
}

void uninstallService(string service_fname)
{
    mixed file_stat;
    string fname = __DIR__ + "i3services/" + service_fname;

    file_stat = stat(fname + ".c");

    if (!objectp(find_object(fname))) return;
    if (arrayp(file_stat) && sizeof(file_stat) && file_stat[2] != 0)
    {
        ANNOUNCE_CHDMOD->rec_m_msg(
          "intermud3",
          "I3",
          "Unloading service '" + service_fname + "'"
        );
        I3LOG("Notice", "I3", "uninstallService", "Unloading service '" + service_fname + "'");
        destruct(find_object(fname));
    }
}

void uninstallServices()
{
    foreach(string fname in get_dir(__DIR__ + "i3services/mod_*.c"))
        uninstallService(fname[0..<3]);
}

void add_service_name(string name, mixed flag)
{
    data["services"][name] = flag;
    save_object(I3_DATA_FILE);
}

void remove_service_name(string name)
{
    map_delete(data["services"], name);
    save_object(I3_DATA_FILE);
}

void add_service(string type, string callback)
{
    services[type] = ([
      "owner": previous_object(),
      "func": callback,
    ]);
}

void remove_service(string type) { map_delete(services, type); }

void setup_services()
{
    data["services"] = ([ ]);
    data["services"]["http"] = port() + 1;
    installServices();
}

void setup_other()
{
    mixed array tm;
    string tm_str, ord;
    int tenth;

    tm = localtime(time() - uptime());
    tenth = tm[LT_MDAY] % 10;

    if (tm[LT_MDAY] > 3 && tm[LT_MDAY] < 21) ord = "th";
    else if (tenth > 3) ord = "th";
    else ord = SpecialOrdinals[tenth];

    tm_str = sprintf("%s, %d%s %s %4d - %02d:%02d:%02d %s",
      Days[tm[LT_WDAY]],
      tm[LT_MDAY],
      ord,
      Months[tm[LT_MON]],
      tm[LT_YEAR],
      tm[LT_HOUR],
      tm[LT_MIN],
      tm[LT_SEC],
      tm[LT_ZONE]
    );

    /* OTHER SETUP HERE */
    /* Ensure they get added to (mapping) data["other"] */
    data["other"] = ([
/*
      Until I figure a way to get the host name, this is commented out.

      "url": "http://ebspso.dnsalias.org:" + (port() + 1) + "/",
*/
      "uptime": I3_MUD_NAME + " has been running since " + tm_str,
      "oob tcp port": "" + OOB_TCP_PORT,
    ]);
}

int setup_i3_socket()
{
    object sock = find_object(SOCKET_D);
    string router_ip;
    int router_port;

    router_ip = explode(data["router_list"][0][1], " ")[0];
    router_port = to_int(explode(data["router_list"][0][1], " ")[1]);

    sock->set_socket_type(MUD);
    sock->set_read_cb("read_callback");
    sock->set_close_cb("close_callback");
    sock->set_log_file(LOG_I3);

    return sock->client_create("I3", router_ip, router_port);
}

void connect()
{
    I3LOG("Notice", "I3", "connect", "Sending connection details.");

    idleTimeout = time();

    sendToRouter("startup-req-3", 0, ({
      data["router_password"],
      data["mudlist_id"],
      data["chanlist_id"],

      /* These correspond to the values in a mudlist info_mapping */
      port(),
      OOB_TCP_PORT,
      0,
      lib_name() + " " + lib_version(),
      baselib_name() + " " + baselib_version(),
      driver_version(),
      "LPMud",
      open_status(),
      admin_email(),
      data["services"],
      data["other"]
    }) );

    save_object(I3_DATA_FILE);
}

varargs void reconnect(int wait)
{
    int x;

    if(!undefinedp(wait))
    {
        call_out("reconnect", wait);
        return;
    }

    initData();
    addRouter(I3_DEF_ROUTER_NAME, I3_DEF_ROUTER_IP, I3_DEF_ROUTER_PORT);

    if(file_exists(I3_DATA_FILE + ".o")) restore_object(I3_DATA_FILE);

    data["reconnect_wait"] += RETRY_TIME_STEP;

    if(data["reconnect_wait"] > MAX_RETRY_TIME) data["reconnect_wait"] = MAX_RETRY_TIME;

    if((x = setup_i3_socket()) < 0)
    {

        if(x == EEISCONN)
        {
            object sock = find_object(SOCKET_D);

            /* Close the socket. */
            sock->close("I3");
            data["reconnect_wait"] -= RETRY_TIME_STEP;
            save_object(I3_DATA_FILE);
            call_out("reconnect", 5);

            return;
        }

        I3LOG("Error", "I3", "reconnect", "Unable to setup socket - " + socket_error(x));

        destruct();

        return;
    }

    setup_services();
    setup_other();

    call_out("connect", 2);

    save_object(I3_DATA_FILE);
}

void shutDown(int restart_delay)
{
    object sock = find_object(SOCKET_D);

    setData("routerConnected", 0);
    uninstallServices();

    save_object(I3_DATA_FILE);

    /* Tell the router we are shutting down */
    sendToRouter("shutdown", 0, ({ restart_delay }) );

    /* Close the socket. */
    sock->close("I3");

    I3LOG("Success", "I3", "shutDown", "Intermud3 Communications shutdown.");
}

void close_callback(int fd)
{
    setData("routerConnected", 0);
    uninstallServices();

    save_object(I3_DATA_FILE);

    foreach(mixed callout in call_out_info())
    {

        if(callout[0] == this_object())
            remove_call_out(callout[1]);

    }

    if(file_exists(I3_CHANLIST_FILE + ".o")) rm(I3_CHANLIST_FILE + ".o");
    if(file_exists(I3_MUDLIST_FILE + ".o")) rm(I3_MUDLIST_FILE + ".o");

    I3LOG("Notice", "I3", "close_callback", "Re-establishing connection in " + data["reconnect_wait"] + " seconds.");
    call_out("reconnect", data["reconnect_wait"]);
}

void read_callback(int fd, mixed packet)
{
    object owner = this_object();
    string err, i3Type, func = "dummy";

    if (!sizeof(packet)) return;

    if (!arrayp(packet)) err = "packet not array";
    else if (sizeof(packet) <= I3_T_USER) err = "packet size too small";
    else if (stringp(packet[I3_T_MUD])
          && packet[I3_T_MUD] != I3_MUD_NAME
          && packet[I3_TYPE] == "mudlist")
    {
        I3LOG(
          "Warning",
          "I3",
          "read_callback",
          "Wrong destination (" + packet[I3_T_MUD] + ") for mudlist packet."
        );
        packet[I3_T_MUD] = mud_name();
    }
    else if (stringp(packet[I3_T_MUD]) && packet[I3_T_MUD] != I3_MUD_NAME)
        err = "wrong destination mud";
    else if (!stringp(packet[I3_O_MUD])) err = "originating mud not a string";
    else if (!stringp(packet[I3_TYPE])) err = "SERVICE is not a string";

    if (err)
    {
        I3LOG("Error", "I3", "read_callback", err + ".");
        I3LOG("Error", "I3", "read_callback", sprintf("%O", packet));

        return;
    }

    setData("routerConnected", 1);
    idleTimeout = time();
    i3Type = packet[I3_TYPE];

    if (i3Type == "mudlist")
    {
#if 1
        I3LOG("Notice", "I3", "read_callback", "Received 'mudlist'");
#else
        I3LOG("Notice", "I3", "read_callback", sprintf("Received %O", packet));
#endif
    }
    else
    {
        I3LOG("Notice", "I3", "read_callback", sprintf("Received %O", packet));
    }

    /* Sanity check on the originator username */
    if (stringp(packet[I3_O_USER]))
        packet[I3_O_USER] = lower_case(packet[I3_O_USER]);

    /* Sanity check on the target username */
    if (stringp(packet[I3_T_USER]))
        packet[I3_T_USER] = lower_case(packet[I3_T_USER]);

    if (member_array(i3Type, keys(services)) != -1)
    {

        if (packetHook[i3Type])
            foreach (string owner_fn, func in packetHook[i3Type])
            {
                owner = find_object(owner_fn);

                if (objectp(owner) && call_other(owner, func, packet)) return;
            }

        owner = services[i3Type]["owner"];
        func = services[i3Type]["func"];
    }

    if (function_exists(func, owner))
        call_other(owner, func, packet);
    else
        call_other(this_object(), "i3_rec_unsupported", packet);

}

void i3_rec_unsupported(mixed *packet)
{
    ANNOUNCE_CHDMOD->rec_m_msg(
      "intermud3",
      "I3",
      sprintf(
        "Service handler for I3 packet %s%O%s not available.",
        HIR,
        packet,
        NOR
      )
    );
    I3LOG(
      "Fatal",
      "I3",
      "unsupported",
      sprintf("Service handler for I3 packet %O not available.", packet)
    );
}
