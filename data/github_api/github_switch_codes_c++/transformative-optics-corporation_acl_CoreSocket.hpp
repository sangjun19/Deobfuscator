/**
 *    \copyright Copyright 2021 Aqueti, Inc. All rights reserved.
 *    \license This project is released under the MIT Public License.
**/

#pragma once
#include <cstdlib>
#include <cstdint>

//=======================================================================
// Figure out whether we're using Windows sockets or not.

// let's start with a clean slate
#undef ACL_USE_WINSOCK_SOCKETS

// Does cygwin use winsock sockets or unix sockets?  Define this before
// compiling the library if you want it to use WINSOCK sockets.
//#define CYGWIN_USES_WINSOCK_SOCKETS

#if defined(_WIN32) && (!defined(__CYGWIN__) || defined(CYGWIN_USES_WINSOCK_SOCKETS))
  #define ACL_USE_WINSOCK_SOCKETS
#endif

//=======================================================================
// Architecture-dependent include files.

#ifndef ACL_USE_WINSOCK_SOCKETS
#include <sys/time.h>   // for timeval, timezone, gettimeofday
#include <sys/select.h> // for fd_set
#include <netinet/in.h> // for htonl, htons
#include <poll.h>       // for poll()
#endif

#ifdef ACL_USE_WINSOCK_SOCKETS
  // These are a pair of horrible hacks that instruct Windows include
  // files to (1) not define min() and max() in a way that messes up
  // standard-library calls to them, and (2) avoids pulling in a large
  // number of Windows header files.  They are not used directly within
  // the Sockets library, but rather within the Windows include files to
  // change the way they behave.

#ifndef NOMINMAX
#define ACL_CORESOCKET_REPLACE_NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define ACL_CORESOCKET_REPLACE_WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h> // struct timeval is defined here
#ifdef ACL_CORESOCKET_REPLACE_NOMINMAX
#undef NOMINMAX
#endif
#ifdef ACL_CORESOCKET_REPLACE_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#endif

#endif

//=======================================================================
// All externally visible symbols should be defined in the name space.

namespace acl { namespace CoreSocket {

//=======================================================================
// Architecture-dependent definitions.

#ifndef ACL_USE_WINSOCK_SOCKETS

  // On Winsock, we have to use SOCKET, so we're going to have to use it
  // everywhere.
  typedef int SOCKET;
  // On Winsock, INVALID_SOCKET is #defined as ~0 (sockets are unsigned ints)
  // We can't redefine it locally, so we have to switch to another name
  static const int BAD_SOCKET = -1;
#else // winsock sockets

  // Bring the SOCKET type into our namespace, basing it on the root namespace one.
  typedef SOCKET SOCKET;

  // Make a namespaced INVALID_SOCKET definition, which cannot be just
  // INVALID_SOCKET because Windows #defines it, so we pick another name.
  static const SOCKET BAD_SOCKET = INVALID_SOCKET;
#endif

/**
 *      This routine will write a block to a file descriptor.  It acts just
 * like the write() system call does on files, but it will keep sending to
 * a socket until an error or all of the data has gone.
 *      This will also take care of problems caused by interrupted system
 * calls, retrying the write when they occur.  It will also work when
 * sending large blocks of data through socket connections, since it will
 * send all of the data before returning.
 *	This routine will return the number of bytes written (which may be
* less than requested in case of EOF), or -1 in the case of an error.
 */

int noint_block_write(SOCKET outsock, const char* buffer, size_t length);

/**
 * @brief Read the specified number of bytes, retrying in case of interrupts.
 *
 *      This routine will read in a block from the file descriptor.
 * It acts just like the read() routine does on normal files, so that
 * it hides the fact that the descriptor may point to a socket.
 *      This will also take care of problems caused by interrupted system
 * calls, retrying the read when they occur.
 *	This routine will either read the requested number of bytes and
 * return that or return -1 (in the case of an error or EOF being reached
 * before all the data arrives).
 * @param [in] insock Socket to read from
 * @param [out] buffer Pointer to the buffer to write the data to.  The client
 *          must have allocated at least length characters here.
 * @param [in] length How many characters to read.
 */

int noint_block_read(SOCKET insock, char* buffer, size_t length);

/**
 *	This routine will perform like a normal select() call, but it will
 * restart if it quit because of an interrupt.  This makes it more robust
 * in its function, and allows this code to perform properly on pxpl5, which
 * sends USER1 interrupts while rendering an image.
 */
int noint_select(int width, fd_set* readfds, fd_set* writefds,
	fd_set* exceptfds, struct timeval* timeout);

/**
 *   This routine will read in a block from the file descriptor.
 * It acts just like the read() routine on normal files, except that
 * it will time out if the read takes too long.
 *   This will also take care of problems caused by interrupted system
 * calls, retrying the read when they occur.
 *   This routine will either read the requested number of bytes and
 * return that or return -1 (in the case of an error or in the case
 * of EOF being reached before all the data arrives), or return the number
 * of characters read before timeout (in the case of a timeout).
 */

int noint_block_read_timeout(SOCKET insock, char* buffer, size_t length,
	struct timeval* timeout);

/**
 * @brief Poll for accept on a socket that is listening.
 *
 * This routine will check the listen socket to see if there has been a
 * connection request. If so, it will accept a connection on the accept
 * socket and set TCP_NODELAY on that socket. The attempt will timeout
 * in the amount of time specified.  
 * @param [in] listen_sock The socket that is listening.
 * @param [out] accept_sock Pointer to the socket to be filled in if
 *          an accept was made.
 * @param [in] timeout How long to wait in seconds before giving up and
 *          returning.
 * @return If the accept and set are successful, it returns 1. If there
 * is nothing asking for a connection, it returns 0. If there is an error
 * along the way, it returns -1.
 */

int poll_for_accept(SOCKET listen_sock, SOCKET* accept_sock,
	double timeout = 0.0);

/**
	* @brief Opens a socket with the requested port number and network interface.
  *
  * @param [in] type The type of socket: SOCK_DGRAM (udp) or SOCK_STREAM (tcp).
  * @param [inout] portno A pointer to a value containing the port to open,
  *           a Null pointer or a pointer to 0 means "any port", and a pointer
  *           to a number specifies that port.  If a pointer to 0 is passed in,
  *           the actual port opened will be filled into on successful return.
  * @param [inout] IPaddress A pointer to the dotted-decimal or DNS name of
  *           one of the network interfaces associated with this computer. A
  *           Null pointer or INADDR_ANY (a pointer to an empty string) uses the
  *           default interface.  A non-empty name will select a particular
  *           interface.
  * @param [in] reuseAddr Forcibly bind to a port even if it is already open
  *           by another application?  This is useful when there is a zombie
  *           server on a well-known port and you're trying to re-open that
  *           port as its replacement.
  * @return BAD_SOCKET on failure and the socket identifier on success.
	*/

SOCKET open_socket(int type, unsigned short* portno, const char* IPaddress,
  bool reuseAddr = false);

/**
  * @brief Opens a UDP socket with the requested port number and network interface.
  *
  * @param [inout] portno A pointer to a value containing the port to open,
  *           a Null pointer or a pointer to 0 means "any port", and a pointer
  *           to a number specifies that port.  If a pointer to 0 is passed in,
  *           the actual port opened will be filled into on successful return.
  * @param [inout] IPaddress A pointer to the dotted-decimal or DNS name of
  *           one of the network interfaces associated with this computer. A
  *           Null pointer or INADDR_ANY (a pointer to an empty string) uses the
  *           default interface.  A non-empty name will select a particular
  *           interface.
  * @param [in] reuseAddr Forcibly bind to a port even if it is already open
  *           by another application?  This is useful when there is a zombie
  *           server on a well-known port and you're trying to re-open that
  *           port as its replacement.
  * @return BAD_SOCKET on failure and the socket identifier on success.
  */

SOCKET open_udp_socket(unsigned short* portno = nullptr, const char* IPaddress = nullptr,
  bool reuseAddr = false);

/**
  * @brief Opens a TCP socket with the requested port number and network interface.
  *
  * @param [inout] portno A pointer to a value containing the port to open,
  *           a Null pointer or a pointer to 0 means "any port", and a pointer
  *           to a number specifies that port.  If a pointer to 0 is passed in,
  *           the actual port opened will be filled into on successful return.
  * @param [inout] IPaddress A pointer to the dotted-decimal or DNS name of
  *           one of the network interfaces associated with this computer. A
  *           Null pointer or INADDR_ANY (a pointer to an empty string) uses the
  *           default interface.  A non-empty name will select a particular
  *           interface.
  * @param [in] reuseAddr Forcibly bind to a port even if it is already open
  *           by another application?  This is useful when there is a zombie
  *           server on a well-known port and you're trying to re-open that
  *           port as its replacement.
  * @return BAD_SOCKET on failure and the socket identifier on success.
  */

SOCKET open_tcp_socket(unsigned short* portno = NULL, const char* NIC_IP = NULL,
  bool reuseAddr = false);

/**
  * @brief Sets options on the specified TCP socket.
  *
  * @param [in] s Socket to set the options on.
  * @param [in] options Set of options for the socket that defaults to Aqueti
  *           default values.
  * @return False on failure to set any of the options, true on success.
  */

/// @brief Options passed to option-setting routine.
class TCPOptions {
public:
  // These are the defaults for an Aqueti connection.
  bool keepAlive = true;
  int keepCount = 4;
  int keepIdle = 20;
  int keepInterval = 5;
  unsigned userTimeout = 15000;
  bool nodelay = true;
  bool ignoreSIGPIPE = true;

  // These are the system defaults
  void UseSystemDefaults() {
    keepAlive = false;
    keepCount = -1;
    keepIdle = -1;
    keepInterval = -1;
    userTimeout = 0;
    nodelay = false;
    ignoreSIGPIPE = false;
  }
};

bool set_tcp_socket_options(SOCKET s, TCPOptions options = TCPOptions());

/**
 * Create a UDP socket and connect it to a specified port.
 */

SOCKET connect_udp_port(const char* machineName, int remotePort,
	const char* NIC_IP = NULL);

/**
 * Retrieves the IP address or hostname of the local interface used to connect
 * to the specified remote host.
 * XXX: This does not always work.  See the Github issue with the report from
 * Isop W. Alexander showing that a machine with two ports (172.28.0.10 and
 * 192.168.191.130) sent a connection request that came from the 172 IP address
 * but that had the name of the 192 interface in the message as the host to
 * call back.  This turned out to be unroutable, so the server failed to call
 * back on the correct IP address.  Presumably, this happens when the gateway
 * is configured to be a single outgoing NIC.  This was on a Linux box.  We
 * need a more reliable way to select the outgoing NIC.  XXX Actually, the
 * problem may be that the UDP receipt code may use the IP address the message
 * came from rather than the machine name in the message.
 *
 * @param local_host A buffer of size 64 that will contain the name of the local
 * interface.
 * @param max_length The maximum length of the local_host buffer.
 * @param remote_host The name of the remote host.
 *
 * @return Returns -1 on getsockname() error, or the output of sprintf
 * building the local_host string.
 */
int get_local_socket_name(char* local_host, size_t max_length,
	const char* remote_host);

/**
 * @brief Get a TCP socket that is ready to accept connections.
 *
 * Ready to accept means that listen() has already been called on it.
 * It can be asked for a specific port or to get whatever port is
 * available from the system. On success, it fills in the pointers to
 * the socket and the port number of the socket that it obtained.
 * @param [inout] listen_portnum The port that the socket is listening on.
 *          If this is a pointer to a nonzero value, the function will
 *          attempt to open a socket on the specified port; if it is a
 *          pointer to 0 then it will open any available port.
 * @param [in] NIC_IP Name or dotted-decimal IP address of the network
 *          interface to use.  The default Null pointer means "listen on all".
 * @param [in] reuseAddr Forcibly bind to a port even if it is already open
 *           (but not bound for accept) by another application.  This is useful
 *           when there is a zombie server on a well-known port and you're
 *           trying to re-open that port as its replacement.
 * @param [in] backlog How many connections can be pending before new ones
 *           are rejected.
 * @param [in] options TCP socket options to set before calling listen()
 *           on the socket.  If nullptr, nothing is set.
 * @return Opened socket on success, BAD_SOCKET on failure.
 */

acl::CoreSocket::SOCKET get_a_TCP_socket(int* listen_portnum,
	const char* NIC_IP = NULL, int backlog = 1000,
  bool reuseAddr = false, const acl::CoreSocket::TCPOptions *options = nullptr);

/// @brief Open a client TCP socket and connect it to a server on a known port
/// @param [in] DNS name or dotted-decimal IP name of the host to connect to.
/// @param [in] port The port to connect to.
/// @param [in] NICaddress Name of the network card to use, can be obtained
///             by calling getmyIP() or set to NULL to use the default network card.
/// @param [out] s Pointer to be filled in with the socket that is connected.
/// @param [in] options TCP options to be set on the socket before connect()
///             is called.  If this is Null, do not set options.
/// @return True on success, false on failure.
bool connect_tcp_to(const char* addr, int port, const char *NICaddress, SOCKET *s,
        const acl::CoreSocket::TCPOptions *options = nullptr);

/// @brief Close a socket.
/// @param [in] Socket descriptor returned by open_socket() or one of the routines
///         that call it; open_udp_socket() or open_tcp_socket().
/// @return 0 on success, nonzero on failure, -100 if sock is BAD_SOCKET.
int close_socket(SOCKET sock);

/// @brief Disables sends or receives on a socket.  Does not need to be called
///        before close_socket(), which will also do this.
/// @param [in] Socket descriptor returned by open_socket() or one of the routines
///         that call it; open_udp_socket() or open_tcp_socket().
/// @return 0 on success, nonzero on failure, -100 if sock is BAD_SOCKET.
int shutdown_socket(SOCKET sock);

/// @brief Cause a TCP socket to accumulate data but not to send it.
///
/// On Windows, this has the side effect of disabling TCP_NODELAY on the socket.
bool cork_tcp_socket(SOCKET sock);

/// @brief Cause a TCP socket to send all accumulated data immediately.
///
/// On Windows, this has the side effect of enabling TCP_NODELAY on the socket.
bool uncork_tcp_socket(SOCKET sock);

/// @brief Helper function that determines whether the socket is ready to read.
///
/// Note that for a socket that is in the listen state then ready to read
/// indicates that it is ready to call accept() on.
/// @param [in] s Socket to check
/// @param [in] timeout Time in seconds to wait until giving up.
/// @return -1 on error or socket exception, 0 on timeout, 1 if there is data
///         ready to ready (or the socket is ready to accept a connection).
int check_ready_to_read_timeout(SOCKET s, double timeout);

/// @brief Convert types to and from network-standard byte order.
double hton(double d);
double ntoh(double d);

int64_t hton(int64_t d);
int64_t ntoh(int64_t d);

}  }	// End of namespace definitions.
