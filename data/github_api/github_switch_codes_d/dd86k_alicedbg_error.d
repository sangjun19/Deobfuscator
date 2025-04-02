// Repository: dd86k/alicedbg
// File: src/adbg/error.d

/// Error handling module.
///
/// NOTE: Every thing that could go wrong should have an error code.
/// Authors: dd86k <dd@dax.moe>
/// Copyright: © dd86k <dd@dax.moe>
/// License: BSD-3-Clause-Clear
module adbg.error;

version (Windows) {
	import core.sys.windows.winbase : GetLastError, FormatMessageA,
		FORMAT_MESSAGE_FROM_SYSTEM, FORMAT_MESSAGE_MAX_WIDTH_MASK;
	enum ADBG_OS_ERROR_FORMAT = "%08X"; /// Error code format
	enum ERR_OSFMT = "%#x"; /// Error code format
} else {
	enum ADBG_OS_ERROR_FORMAT = "%d"; /// Error code format
	enum ERR_OSFMT = "%d"; /// Error code format
}
import core.stdc.errno : errno;
import core.stdc.string : strerror;
import adbg.include.capstone : csh, cs_errno, cs_strerror;

// TODO: Make module thread-safe
//       Either via TLS and/or atomic operations
// TODO: Error utils
//       adbg_ensure_params(lvalue, "name")
//       - returns string if null found
//       - automatically set error code
//       adbg_oops_ptr(AdbgError, void*) to return null
// TODO: Localize error messages as option (including system ones, when able)

extern (C):

/// Error codes.
enum AdbgError {
	//
	// 0-99: Generic
	//
	success	= 0,
	invalidArgument	= 1,	/// Argument is null or zero
	emptyArgument	= 2,	/// Argument contains an empty dataset
	uninitiated	= 4,	/// Instance was not initiated
	invalidOption	= 5,	/// Invalid option
	invalidValue	= 6,	/// Invalid value for option
	offsetBounds	= 7,	/// File offset is outside of file size
	indexBounds	= 8,	/// Index is outside of bounds of list
	unavailable	= 9,	/// Feature or item is unavailable
	unfindable	= 10,	/// Item not found
	partialRead	= 11,	/// Not all data could be read
	partialWrite	= 12,	/// Not all data could be written
	//
	// 100-199: Debugger
	//
	debuggerUnattached	= 100,
	debuggerUnpaused	= 101,
	debuggerInvalidAction	= 102,	/// Wrong action from creation method
	debuggerPresent	= 103,	/// Debugger already present in remote process
	debuggerNeedFile	= 104,	/// File path not given (e.g., directory)
	//
	// 200-299: Disasembler
	//
	disasmUnsupportedMachine	= 202,
	disasmIllegalInstruction	= 220,
	disasmEndOfData	= 221,
	disasmOpcodeLimit	= 221,
	//
	// 300-399: Object server
	//
	objectUnknownFormat	= 301,
	objectUnsupportedFormat	= 302,
	objectTooSmall	= 303,
	objectMalformed	= 304,
	objectItemNotFound	= 305,
	objectInvalidVersion	= 310,
	objectInvalidMachine	= 311,
	objectInvalidClass	= 312,
	objectInvalidEndian	= 313,
	objectInvalidType	= 314,
	objectInvalidABI	= 315,
	//
	// 400-499: System
	//
	systemLoadError	= 402,
	systemBindError	= 403,
	//
	// 800-899: Memory scanner
	//
	scannerDataEmpty	= 800,
	scannerDataLimit	= 801,
	//
	// 1000-1999: Misc
	//
	assertion	= 1000,	/// Soft assert
	unimplemented	= 1001,	/// Not implemented
	//
	// 2000-2999: External resources
	//
	os	= 2001,
	crt	= 2002,
	//
	// 3000-3999: External libraries
	//
	libCapstone	= 3002,	/// Capstone
}

/// Represents an error in alicedbg.
private
struct adbg_error_t {
	int srccode;	/// Error code from Alicedbg
	int modcode;	/// Error code for module
	const(char)* func;	/// Source function
	int line;	/// Line source
}
/// Last error in alicedbg.
private __gshared adbg_error_t error;

/// Get last Alicedbg error code.
/// Returns: Error code (AdbgError).
int adbg_error_code() {
	return error.srccode;
}
/// Get the last error from the associated module.
/// Returns: Error code (submodule).
int adbg_error_code_external() {
	return error.modcode;
}
/// Get the line of the error code initiator.
/// Returns: Line number.
int adbg_error_line() {
	return error.line;
}
/// Get the function of the error code initiator.
/// Returns: Function name.
const(char)* adbg_error_function() {
	return error.func;
}

/// Get error message from the OS (or CRT) by providing the error code
/// Params: code = Error code number from OS
/// Returns: String
private
const(char)* adbg_error_system_message(int code) {
	version (Windows) {
		//TODO: Handle NTSTATUS codes
		enum ERR_BUF_SZ = 256;
		__gshared char [ERR_BUF_SZ]buffer = void;
		size_t len = FormatMessageA(
			FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_MAX_WIDTH_MASK,
			null,
			code,
			0,	// Default
			buffer.ptr,
			ERR_BUF_SZ,
			null);
		return len ? cast(char*)buffer : "Unknown error";
	} else {
		return strerror(code);
	}
}

//
// ANCHOR Error setters
//

/// Reset the last set error code.
void adbg_error_reset() {
	// NOTE: Code is enough. Other fields are purely internal.
	error.srccode = 0;
}

// NOTE: D compilers changed how __MODULE__ and __FILE__ are evaluated.
//       It used so that the caller evaluated those but now the front-end
//       puts in the value of the callee instead. To prove this, __LINE__
//       and __FUNCTION__ remains unchanged. To fix that, I'm supposed to
//       use a template, but function templates pollute the final binary
//       with instances that has no right to be duplicated.

private
void adbg_error_set(AdbgError e, void *handle, const(char)* func, int line) {
	error.func = func;
	error.line = line;
	// To avoid additional errors, such as formatting,
	// get the underlying error code now for later.
	switch (error.srccode = e) {
	case AdbgError.os:
		version (Windows)
			error.modcode = GetLastError();
		else
			error.modcode = errno;
		version(Trace) trace("oscode="~ERR_OSFMT, error.modcode);
		break;
	case AdbgError.crt:
		error.modcode = errno;
		version(Trace) trace("crt=%d", error.modcode);
		break;
	case AdbgError.libCapstone:
		assert(handle, "oops, no handles");
		error.modcode = cs_errno(cast(csh)handle);
		version(Trace) trace("capstone_error=%d", error.modcode);
		break;
	default:
		error.modcode = 0;
	}
}

/// Internal: Sets the last error code.
/// Params:
/// 	e = Error code.
/// 	handle = External resource (handle, code, etc.).
/// 	func = Automatically set to `__FUNCTION__`.
/// 	line = Automatically set to `__LINE__`.
/// Returns: Error code.
int adbg_oops(AdbgError e, void *handle = null,
	const(char)* func = __FUNCTION__.ptr, int line = __LINE__) {
	version(Trace) trace("code=%d handle=%p caller=%s@%d", e, handle, func, line);
	adbg_error_set(e, handle, func, line);
	return e;
}

/// Internal: Same as adbg_oops, but returns a null pointer.
/// Params:
/// 	e = Error code.
/// 	handle = External resource (handle, code, etc.).
/// 	func = Automatically set to `__FUNCTION__`.
/// 	line = Automatically set to `__LINE__`.
/// Returns: Null pointer.
void* adbg_oops_null(AdbgError e, void *handle = null,
	const(char)* func = __FUNCTION__.ptr, int line = __LINE__) {
	version(Trace) trace("code=%d handle=%p caller=%s@%d", e, handle, func, line);
	adbg_error_set(e, handle, func, line);
	return null;
}

private struct adbg_error_msg_t {
	int code;
	string msg;
}
private immutable const(char) *defaultMsg = "Unknown error occured.";
private immutable adbg_error_msg_t[] errors_msg = [
	//
	// Generics
	//
	{ AdbgError.invalidArgument,	"Invalid or missing parameter value." },
	{ AdbgError.emptyArgument,	"Parameter is empty." },
	{ AdbgError.uninitiated,	"Instance requires to be initialized first." },
	{ AdbgError.invalidOption,	"Unknown option." },
	{ AdbgError.invalidValue,	"Received invalid value for option." },
	{ AdbgError.offsetBounds,	"Offset outside range." },
	{ AdbgError.indexBounds,	"Index outside range." },
	{ AdbgError.unavailable,	"Feature or item is unavailable." },
	{ AdbgError.unfindable,	"Item was not found." },
	{ AdbgError.partialRead,	"Not all required data could be read." },
	{ AdbgError.partialWrite,	"Not all required data could be written." },
	//
	// Debugger
	//
	{ AdbgError.debuggerUnattached,	"Debugger needs to be attached for this feature." },
	{ AdbgError.debuggerUnpaused,	"Debugger needs a stopped process for this feature." },
	{ AdbgError.debuggerInvalidAction,	"Debugger was given a wrong action for this process." },
	{ AdbgError.debuggerPresent,	"Debugger already present on remote process." },
	{ AdbgError.debuggerNeedFile,	"Debugger received a path that does not point to a file." },
	//
	// Disassembler
	//
	{ AdbgError.disasmUnsupportedMachine,	"Disassembler does not support this platform." },
	{ AdbgError.disasmIllegalInstruction,	"Disassembler met an illegal instruction." },
	{ AdbgError.disasmEndOfData,	"Disassembler reached end of data." },
	{ AdbgError.disasmOpcodeLimit,	"Disassembler reached architectural opcode limit." },
	//
	// Object server
	//
	{ AdbgError.objectUnknownFormat,	"Object format unknown." },
	{ AdbgError.objectUnsupportedFormat,	"Object format unsupported." },
	{ AdbgError.objectTooSmall,	"Object is too small to be valid." },
	{ AdbgError.objectMalformed,	"Object potentially corrupted." },
	{ AdbgError.objectItemNotFound,	"Object item was not found." },
	{ AdbgError.objectInvalidVersion,	"Object has invalid version." },
	{ AdbgError.objectInvalidMachine,	"Object has invalid machine or platform value." },
	{ AdbgError.objectInvalidClass,	"Object has invalid class or bitness value." },
	{ AdbgError.objectInvalidEndian,	"Object has invalid endian value." },
	{ AdbgError.objectInvalidType,	"Object type invalid." },
	{ AdbgError.objectInvalidABI,	"Object has Invalid ABI value." },
	//
	// Symbols
	//
	{ AdbgError.systemLoadError,	"Dynamic library could not be loaded." },
	{ AdbgError.systemBindError,	"Symbol could not be binded." },
	//
	// Memory module
	//
	{ AdbgError.scannerDataEmpty,	"Memory scanner received empty data." },
	{ AdbgError.scannerDataLimit,	"Memory scanner received too much data." },
	//
	// Misc.
	//
	{ AdbgError.assertion,	"A soft debug assertion was hit." },
	{ AdbgError.unimplemented,	"Feature is not implemented." },
	{ AdbgError.success,	"No errors occured." },
];

/// Get the last set error message.
/// Returns: Error message.
export
const(char)* adbg_error_message() {
	switch (error.srccode) with (AdbgError) {
	case crt:
		return strerror(error.modcode);
	case os:
		return adbg_error_system_message(error.modcode);
	case libCapstone:
		return cs_strerror(error.modcode);
	default:
		foreach (ref e; errors_msg)
			if (error.srccode == e.code)
				return e.msg.ptr;
	}
	return defaultMsg;
}

version (Trace) {
	import core.stdc.stdio, core.stdc.stdarg;
	private import adbg.include.d.config : D_FEATURE_PRAGMA_PRINTF;
	
	private extern (C) int putchar(int);
	
	static if (D_FEATURE_PRAGMA_PRINTF) {
		/// Trace application
		pragma(printf)
		void trace(string func = __FUNCTION__, int line = __LINE__)(const(char) *fmt, ...) {
			va_list va;
			va_start(va, fmt);
			printf("TRACE:%s@%u: ", func.ptr, line);
			vprintf(fmt, va);
			putchar('\n');
		}
	} else {
		/// Trace application
		void trace(string func = __FUNCTION__, int line = __LINE__)(const(char) *fmt, ...) {
			va_list va;
			va_start(va, fmt);
			printf("TRACE:%s@%u: ", func.ptr, line);
			vprintf(fmt, va);
			putchar('\n');
		}
	}
}
