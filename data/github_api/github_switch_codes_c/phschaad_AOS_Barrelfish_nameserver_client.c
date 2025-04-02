#include <aos/aos.h>
#include <nameserver.h>
#include <aos/nameserver_internal.h> //this is only needed for pretty printing

#include <stdio.h>
#include <stdlib.h>
#include <aos/aos_rpc.h>
#include <aos/generic_threadsafe_queue.h>

struct aos_rpc nameserver_connection;

// COPY FROM AOS_RPC.C FIX IF TIME ALLOWS


struct thread_mutex ns_rpc_mutex;

struct rpc_call {
    unsigned char id;
    unsigned char type;
    bool *done;
    void (*recv_handling)(void *arg1, struct recv_list *data);
    void *arg1;
    struct rpc_call *next;
};

struct rpc_call *ns_rpc_call_buffer_head = NULL;

static void register_recv(unsigned char id, unsigned char type, bool *done,
                          void (*inst_recv_handling)(void *arg1,
                                                     struct recv_list *data),
                          void *recv_handling_arg1)
{
    struct rpc_call *call = malloc(sizeof(struct rpc_call));
    call->id = id;
    call->type = type;
    call->done = done;
    call->recv_handling = inst_recv_handling;
    call->arg1 = recv_handling_arg1;

    // TODO: consider that this might not need a mutex
    synchronized(ns_rpc_mutex)
    {
        call->next = ns_rpc_call_buffer_head;
        ns_rpc_call_buffer_head = call;
    }
}

static void
rpc_framework(void (*inst_recv_handling)(void *arg1, struct recv_list *data),
              void *recv_handling_arg1, unsigned char type,
              struct lmp_chan *chan, struct capref cap, size_t payloadsize,
              uintptr_t *payload, struct event_closure callback_when_done)
{
    unsigned char id = request_fresh_id(RPC_MESSAGE(type));

    bool done = false;
    register_recv(id, RPC_ACK_MESSAGE(type), &done, inst_recv_handling,
                  recv_handling_arg1);
    send(chan, cap, RPC_MESSAGE(type), payloadsize, payload,
         callback_when_done, id);
    struct waitset *ws = get_default_waitset();
//if(strcmp(disp_name(),"init") != 0) //in an ideal world our processes would all have a main loop they are pumping in a seperate thread which only does that pumping and this hack here would not be required, but this world is not that world.
    while (!done)
        event_dispatch(ws);
}


static void ns_rpc_recv_handler(struct recv_list *data)
{
    DBG(VERBOSE,"ns_rpc_recv_handler, raw type %u, id %u\n",(unsigned int)data->type,(unsigned int)data->id);
    // do actions depending on the message type
    // Check the message type and handle it accordingly.
    if (data->type & 1) { // ACK, so we check the recv list
        struct rpc_call *foodata = NULL;
        // TODO: consider that this might not need a mutex lock
        synchronized(ns_rpc_mutex)
        {
            struct rpc_call *prev = NULL;
            foodata = ns_rpc_call_buffer_head;
            while (foodata != NULL) {
                if (foodata->type == data->type &&
                    foodata->id == data->payload[0])
                    break;
                prev = foodata;
                foodata = foodata->next;
            }
            if (foodata != NULL) {
                if (prev == NULL) {
                    ns_rpc_call_buffer_head = ns_rpc_call_buffer_head->next;
                } else {
                    prev->next = foodata->next;
                }
            }
        }
        if (foodata == NULL) {
            DBG(WARN, "did not have a NS RPC receiver registered for the ACK - "
                    "type %u id %u\n",
                (unsigned int) data->type, (unsigned int) data->payload[0]);
            DBG(WARN, "Dumping foodata structure: \n");
            foodata = ns_rpc_call_buffer_head;
            int counter = 0;
            while (foodata != NULL) {
                DBG(WARN, "%d - type %u id % u\n", counter,
                    (unsigned int) foodata->type, (unsigned int) foodata->id);
                counter++;
                foodata = foodata->next;
            }
        } else {
            if (foodata->recv_handling != NULL)
                foodata->recv_handling(foodata->arg1, data);
            MEMORY_BARRIER;
            *foodata->done = true;
            MEMORY_BARRIER;
            free(foodata);
        }
    } else {
        switch (data->type) {
            default:
                debug_printf("got message type %d\n", data->type);
                DBG(WARN, "Unable to handle NS RPC-receipt, expect badness!\n");
        }
    }
}

// END COPY FROM AOS_RPC.C


static void ns_handshake_recv_handler(void* arg1,struct recv_list*data) {
    DBG(VERBOSE,"ns_handshake_recv_handler\n");
    nameserver_connection.chan.remote_cap = data->cap;
}

struct capref ns_cap;
bool ns_cap_set = false;
void set_ns_cap(struct capref cap) {
    ns_cap = cap;
    ns_cap_set = true;
}

static void handshake_with_ns(void) {
    thread_mutex_init(&ns_rpc_mutex);
    if(!ns_cap_set)
        CHECK(aos_rpc_get_nameserver(get_init_rpc(), &ns_cap));
    DBG(VERBOSE,"print ns_cap: slot %u, level %u, cnode %u, croot %u\n", (unsigned int) ns_cap.slot, (
                         unsigned int) ns_cap.cnode.level, (unsigned int) ns_cap.cnode.cnode,
                 (unsigned int) ns_cap.cnode.croot);
    init_rpc_client(ns_rpc_recv_handler,&nameserver_connection.chan,ns_cap);
    rpc_framework(ns_handshake_recv_handler,&nameserver_connection.chan.remote_cap,NS_RPC_TYPE_HANDSHAKE,&nameserver_connection.chan,nameserver_connection.chan.local_cap,0,NULL,NULL_EVENT_CLOSURE);
    DBG(VERBOSE,"print remote_cap: slot %u, level %u, cnode %u, croot %u\n", (unsigned int) nameserver_connection.chan.remote_cap.slot, (
                         unsigned int) nameserver_connection.chan.remote_cap.cnode.level, (unsigned int) nameserver_connection.chan.remote_cap.cnode.cnode,
                 (unsigned int) nameserver_connection.chan.remote_cap.cnode.croot);
    nameserver_connection.init = true;
    thread_yield();
}


errval_t remove_self(void) { //convenience function which removes all services this process added
    if(!nameserver_connection.init)
        handshake_with_ns();
    rpc_framework(NULL,NULL,NS_RPC_TYPE_REMOVE_SELF,&nameserver_connection.chan,NULL_CAP,0,NULL,NULL_EVENT_CLOSURE);
    return SYS_ERR_OK;
}
errval_t register_service(struct nameserver_info *nsi) {
    if(!nameserver_connection.init)
        handshake_with_ns();
    char* ser = serialize_nameserver_info(nsi);
    DBG(VERBOSE,"before sending: %s\n",&ser[8]);
    uintptr_t *out;
    size_t outsize;
    convert_charptr_to_uintptr_with_padding_and_copy(ser,strlen(&ser[8])+9,&out,&outsize);
    DBG(VERBOSE,"before sending2: %s\n",(char*)&out[2]);
    free(ser);
    DBG(VERBOSE,"print cap: slot %u, level %u, cnode %u, croot %u\n",(unsigned int)nsi->chan_cap.slot,(
            unsigned int) nsi->chan_cap.cnode.level, (unsigned int) nsi->chan_cap.cnode.cnode, (unsigned int) nsi->chan_cap.cnode.croot);

    //todo: error handling
    rpc_framework(NULL,NULL,NS_RPC_TYPE_REGISTER_SERVICE,&nameserver_connection.chan,nsi->chan_cap,outsize,out,NULL_EVENT_CLOSURE);
    DBG(VERBOSE,"request was '%s' (skipping first 2 unprintable ints) of length %u",(char*)&out[2],outsize);
    free(out);
    DBG(DETAILED,"sent register request\n");
    return SYS_ERR_OK;
}
errval_t deregister(char* name) {
    if(!nameserver_connection.init)
        handshake_with_ns();
    uintptr_t *out;
    size_t outsize;
    convert_charptr_to_uintptr_with_padding_and_copy(name,strlen(name)+1,&out,&outsize);
    rpc_framework(NULL,NULL,NS_RPC_TYPE_DEREGISTER_SERVICE,&nameserver_connection.chan,NULL_CAP,outsize,out,NULL_EVENT_CLOSURE);
    return SYS_ERR_OK;
}

static void lookup_recv_handler(void*arg1, struct recv_list* data) {
    struct nameserver_info**res = (struct nameserver_info**)arg1;
    if(data->size > 1) {
        deserialize_nameserver_info((char *) &data->payload[1], res);
        (*res)->chan_cap = data->cap;
    }else{
        *res = NULL;
    }
}

errval_t lookup(struct nameserver_query* nsq, struct nameserver_info** result) { //first fit
    if(!nameserver_connection.init)
        handshake_with_ns();
    size_t len = 0;
    char *ser = serialize_nameserver_query(nsq,&len);
    uintptr_t *out;
    size_t outsize;
    convert_charptr_to_uintptr_with_padding_and_copy(ser,len+1,&out,&outsize);
    free(ser);
    rpc_framework(lookup_recv_handler,result,NS_RPC_TYPE_LOOKUP,&nameserver_connection.chan,NULL_CAP,outsize,out,NULL_EVENT_CLOSURE);
    free(out);
    return SYS_ERR_OK;
}

static void enumerate_recv_handler(void*arg1, struct recv_list*data) {
    char***res = (char***)arg1;
    if(data->size <= 1) {
        *res = 0;
        return;
    }
    char* pl = (char*)&data->payload[1];//skipping first byte of reply
    char* pl2 = pl;
    int count = 1;
    while(*pl2 != '\0') {
        if(*pl2 == ',')
            count++;
        pl2++;
    }
    char**arr = malloc(sizeof(char*)*(count+1));
    pl2 = pl;
    int index = 0;
    char* temp;
    while(*pl2 != '\0') {
        if(*pl2 == ',') {
            size_t strlen = pl2-pl;
            temp = malloc(strlen+1);
            memcpy(temp,pl,strlen);
            temp[strlen] = '\0';
            arr[index] = temp;
            index++;
            pl = pl2+1;
        }
        pl2++;
    }
    size_t strlen = pl2-pl;
    temp = malloc(strlen+1);
    memcpy(temp,pl,strlen);
    temp[strlen] = '\0';
    arr[index] = temp;
    //debug_printf(arr[0]);
    *res = arr;
    index++;
    arr[index] = 0;
}

errval_t enumerate(struct nameserver_query* nsq, size_t *num, char*** result) { //all hits
    if(!nameserver_connection.init)
        handshake_with_ns();
    size_t len = 0;
    char *ser = serialize_nameserver_query(nsq,&len);
    uintptr_t *out;
    size_t outsize;
    convert_charptr_to_uintptr_with_padding_and_copy(ser,len+1,&out,&outsize);
    free(ser);
    rpc_framework(enumerate_recv_handler,result,NS_RPC_TYPE_ENUMERATE,&nameserver_connection.chan,NULL_CAP,outsize,out,NULL_EVENT_CLOSURE);
    char** temp = *result;
    *num = 0;
    if(temp != NULL) {
        while (*temp != NULL) {
            (*num)++;
            temp++;
        }
    }
    free(out);

    return SYS_ERR_OK;
}
/*
errval_t enumerate_complex(struct nameserver_query* nsq, size_t *num, struct nameserver_info** result) { //all hits, names only
    if(!nameserver_connection.init)
        handshake_with_ns();
    return LIB_ERR_NOT_IMPLEMENTED;
}
*/
void ns_debug_dump(void) {
    if(!nameserver_connection.init)
        handshake_with_ns();
    rpc_framework(NULL,NULL,NS_RPC_TYPE_DEBUG_DUMP,&nameserver_connection.chan,NULL_CAP,0,NULL,NULL_EVENT_CLOSURE);
}