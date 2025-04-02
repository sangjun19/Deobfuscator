/////// /////// /////// /////// /////// /////// ///////
///
///    H
///
/////// (H lang) by @ENDESGA 2024 :::.

/////// /////// /////// /////// /////// /////// ///////
///////  only define once
#ifndef H_LANG
#define H_LANG

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// check compiler

#define COMPILER_INTEL 0
#define COMPILER_TCC 0
#define COMPILER_GCC 0
#define COMPILER_CLANG 0
#define COMPILER_MSVC 0
#define COMPILER_UNKNOWN 0

#if defined( __INTEL_COMPILER )
#undef COMPILER_INTEL
#define COMPILER_INTEL 1
#define COMPILER_NAME "Intel"
#define COMPILER_VERSION_MAJOR ( __INTEL_COMPILER / 100 )
#define COMPILER_VERSION_MINOR ( __INTEL_COMPILER % 100 )
//
#elif defined( __TINYC__ )
#undef COMPILER_TCC
#define COMPILER_TCC 1
#define COMPILER_NAME "TCC"
#define COMPILER_VERSION_MAJOR ( __TINYC__ / 100 )
#define COMPILER_VERSION_MINOR ( __TINYC__ % 100 )
//
#elif defined( __GNUC__ )
#undef COMPILER_GCC
#define COMPILER_GCC 1
#define COMPILER_NAME "GCC"
#define COMPILER_VERSION_MAJOR ( __GNUC__ )
#define COMPILER_VERSION_MINOR ( __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ )
//
#elif defined( __clang__ )
#undef COMPILER_CLANG
#define COMPILER_CLANG 1
#define COMPILER_NAME "Clang"
#define COMPILER_VERSION_MAJOR ( __clang_major__ )
#define COMPILER_VERSION_MINOR ( __clang_minor__ * 100 + __clang_patchlevel__ )
//
#elif defined( _MSC_VER )
#undef COMPILER_MSVC
#define COMPILER_MSVC 1
#define COMPILER_NAME "MSVC"
#define COMPILER_VERSION_MAJOR ( _MSC_VER / 100 )
#define COMPILER_VERSION_MINOR ( _MSC_VER % 100 )
//
#else
#undef COMPILER_UNKNOWN
#define COMPILER_UNKNOWN 1
#define COMPILER_NAME "unknown"
#define COMPILER_VERSION_MAJOR 0
#define COMPILER_VERSION_MINOR 0
#endif

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// sanity check

#if ( COMPILER_INTEL + COMPILER_TCC + COMPILER_GCC + COMPILER_CLANG + \
	COMPILER_MSVC ) != 1
#error "No Compiler Detected."
#endif
//
#if !( defined( __x86_64__ ) || defined( __aarch64__ ) )
#error "Compiling H Requires a 64-bit Compiler."
#endif

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// magic macros

#define EVAL( ... ) __VA_ARGS__
#define JOIN( a, ... ) a##__VA_ARGS__
//
#define SKIP_ARG( ARG, ... ) __VA_ARGS__
//
#define GET_ARG0( ... )
#define GET_ARG1( a, ... ) a
#define GET_ARG2( a, b, ... ) b
#define GET_ARG3( a, ... ) GET_ARG2( __VA_ARGS__ )
#define GET_ARG4( a, ... ) GET_ARG3( __VA_ARGS__ )
#define GET_ARG5( a, ... ) GET_ARG4( __VA_ARGS__ )
#define GET_ARG6( a, ... ) GET_ARG5( __VA_ARGS__ )
#define GET_ARG7( a, ... ) GET_ARG6( __VA_ARGS__ )
#define GET_ARG8( a, ... ) GET_ARG7( __VA_ARGS__ )
#define GET_ARG9( a, ... ) GET_ARG8( __VA_ARGS__ )
#define GET_ARG10( a, ... ) GET_ARG9( __VA_ARGS__ )
#define GET_ARG11( a, ... ) GET_ARG10( __VA_ARGS__ )
#define GET_ARG12( a, ... ) GET_ARG11( __VA_ARGS__ )
#define GET_ARG13( a, ... ) GET_ARG12( __VA_ARGS__ )
#define GET_ARG14( a, ... ) GET_ARG13( __VA_ARGS__ )
#define GET_ARG15( a, ... ) GET_ARG14( __VA_ARGS__ )
#define GET_ARG16( a, ... ) GET_ARG15( __VA_ARGS__ )
#define GET_ARG17( a, ... ) GET_ARG16( __VA_ARGS__ )
//
#define COMMA ,
//
#define _PASTE_IF_ARGS_EVAL( CODE, ... ) \
	EVAL( GET_ARG2( __VA_ARGS__, EVAL CODE ) )
#define _PASTE_IF_ARGS( CODE, ... ) \
	_PASTE_IF_ARGS_EVAL( CODE, GET_ARG1 __VA_ARGS__( COMMA ) )
#define PASTE_IF_ARGS( CODE, ... ) \
	_PASTE_IF_ARGS( ( CODE ), GET_ARG1( __VA_ARGS__ ) )
//
#define COMMA_IF_ARGS( ... ) PASTE_IF_ARGS( COMMA, __VA_ARGS__ )
//
#define _COUNT_ARGS_EVAL( ... ) \
	EVAL( GET_ARG17( \
		__VA_ARGS__ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 ) )
#define COUNT_ARGS( ... ) \
	_COUNT_ARGS_EVAL( __VA_ARGS__ COMMA_IF_ARGS( __VA_ARGS__ ) )

// // // // // // //
// default arguments

#define _DEFAULTS_PARAM_EVAL( DEF, ... ) GET_ARG2( __VA_ARGS__, DEF )
#define _DEFAULTS_PARAM( DEFS, ... ) \
	_DEFAULTS_PARAM_EVAL( \
		GET_ARG1 DEFS, _ COMMA_IF_ARGS( __VA_ARGS__ ) __VA_ARGS__ )
#define _DEFAULTS_0( DEFS, ... )
#define _DEFAULTS_1( DEFS, ... ) _DEFAULTS_PARAM( DEFS, __VA_ARGS__ )
#define _DEFAULTS_2( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_1( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_3( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_2( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_4( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_3( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_5( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_4( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_6( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_5( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_7( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_6( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_8( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_7( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_9( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_8( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_10( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_9( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_11( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_10( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_12( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_11( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_13( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_12( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_14( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_13( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_15( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_14( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
#define _DEFAULTS_16( DEFS, ... ) \
	_DEFAULTS_PARAM( DEFS, __VA_ARGS__ ), \
		_DEFAULTS_15( ( SKIP_ARG DEFS ), SKIP_ARG( __VA_ARGS__ ) )
//
#define _DEFAULTS_MAKE( COUNT, DEFS, ... ) \
	JOIN( _DEFAULTS_, COUNT )( DEFS, __VA_ARGS__ )
#define DEFAULTS( DEFS, ... ) \
	_DEFAULTS_MAKE( COUNT_ARGS DEFS, DEFS, __VA_ARGS__ )

// // // // // // //
// symbol chain arguments

#define _CHAIN_0( BEFORE, AFTER, BETWEEN, ... ) __VA_ARGS__
#define _CHAIN_1( BEFORE, AFTER, BETWEEN, A ) BEFORE A AFTER
#define _CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, B ) \
	_CHAIN_1( BEFORE, AFTER, BETWEEN, A ) BETWEEN B
#define _CHAIN_2( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_1( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_3( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_2( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_4( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_3( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_5( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_4( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_6( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_5( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_7( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_6( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_8( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_7( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_9( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_8( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_10( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_9( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_11( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_10( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_12( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_11( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_13( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_12( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_14( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_13( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_15( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_14( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
#define _CHAIN_16( BEFORE, AFTER, BETWEEN, A, ... ) \
	_CHAIN_JOIN( BEFORE, AFTER, BETWEEN, A, \
		_CHAIN_15( BEFORE, AFTER, BETWEEN, __VA_ARGS__ ) )
//
#define _CHAIN_MAKE( COUNT ) JOIN( _CHAIN_, COUNT )
#define CHAIN( BEFORE, AFTER, BETWEEN, ... ) \
	_CHAIN_MAKE( COUNT_ARGS( __VA_ARGS__ ) ) \
	( BEFORE, AFTER, BETWEEN, __VA_ARGS__ )

// // // // // // //
// struct construction

#define _CONSTRUCT_PARAM( FIELD, VAL ) .FIELD = VAL
#define _CONSTRUCT_0( FIELDS, ... )
#define _CONSTRUCT_1( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) )
#define _CONSTRUCT_2( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_1( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_3( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_2( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_4( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_3( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_5( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_4( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_6( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_5( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_7( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_6( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_8( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_7( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_9( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_8( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_10( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_9( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_11( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_10( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_12( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_11( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_13( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_12( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_14( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_13( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_15( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_14( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
#define _CONSTRUCT_16( FIELDS, ... ) \
	_CONSTRUCT_PARAM( GET_ARG1 FIELDS, GET_ARG1( __VA_ARGS__ ) ), \
		_CONSTRUCT_15( ( SKIP_ARG FIELDS ), SKIP_ARG( __VA_ARGS__ ) )
//
#define _CONSTRUCT_MAKE( COUNT, FIELDS, ... ) \
	{ JOIN( _CONSTRUCT_, COUNT )( FIELDS, __VA_ARGS__ ) }
#define _CONSTRUCT_EXPAND( FIELDS, DEFS, ... ) \
	_CONSTRUCT_MAKE( COUNT_ARGS FIELDS, FIELDS, DEFAULTS( DEFS, __VA_ARGS__ ) )
#define CONSTRUCT( FIELDS, DEFS, ... ) \
	_CONSTRUCT_EXPAND( FIELDS, DEFS, __VA_ARGS__ )

// // // // // // //
// define bounds

#define DEF_START \
	do {
//
#define DEF_END \
	} \
	while( 0 )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// basic type and memory definitions

#define mutable /*explicitly mutable*/
#define mutable_ref *
#define ref mutable_ref const __restrict
#define temp register
#define perm static
#define anon void
#define global /*explicitly global*/

// // // // // // //
// type operations

#define ref_of( ... ) ( &__VA_ARGS__ )
#define val_of( ... ) ( *__VA_ARGS__ )
#define size_of sizeof
#define type_from( ... ) typedef __VA_ARGS__
#define to( TYPE, ... ) ( (TYPE)( __VA_ARGS__ ) )
#define cast( TYPE, ... ) \
	val_of( to( TYPE mutable_ref, ref_of( __VA_ARGS__ ) ) )

// // // // // // //
// function operations and prefixes

#if COMPILER_MSVC
#define _FORCE_INLINE __forceinline
#define PACKED
#else
#define _FORCE_INLINE __attribute__( ( always_inline ) )
#define PACKED __attribute__( ( packed ) )
#endif

#define EMBED perm inline
#define embed EMBED _FORCE_INLINE
#define fn EMBED anon
#define in const
#define out return

#define fn_ref( OUTPUT, NAME, ... ) OUTPUT const val_of( NAME )( __VA_ARGS__ )
#define fn_type( OUTPUT, ... ) \
	type_from( struct { fn_ref( OUTPUT, _FN, __VA_ARGS__ ); }* )

#define fn_type_from( FN ) type_from( type_of( FN ) )
#define set_fn_ref( FN_REF, NEW_FN_REF ) \
	FN_REF = to( type_of( FN_REF ), NEW_FN_REF )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// fundamentals

#define null to( anon ref, 0 )
//
#define not !
#define and &&
#define or ||
#define xor ^
#define mod %
#define is ==
#define isnt !=
#define is_even mod 2 is 0
#define is_odd mod 2 isnt 0

// // // // // // //
// control flow

#define loop for( ;; )
#define leave break
#define skip continue
#define until( ... ) while( not( __VA_ARGS__ ) )

#define select( ... ) switch( __VA_ARGS__ )
#define with( ... ) CHAIN( case, :, , __VA_ARGS__ )
#define otherwise default:

#define _do_once( COUNTER ) \
	perm char JOIN( _ONCE_, COUNTER ) = 1; \
	if( ( JOIN( _ONCE_, COUNTER ) == 1 ? JOIN( _ONCE_, COUNTER )-- : 0 ) )
#define do_once _do_once( __COUNTER__ )

#define pick( IF_YES, THEN_THIS, ELSE_THIS ) \
	( ( IF_YES ) ? ( THEN_THIS ) : ( ELSE_THIS ) )

#define if_all( ... ) if( CHAIN(, , and, __VA_ARGS__ ) )
#define if_any( ... ) if( CHAIN(, , or, __VA_ARGS__ ) )
#define if_not_all( ... ) if( not( CHAIN(, , and, __VA_ARGS__ ) ) )
#define if_not_any( ... ) if( not( CHAIN(, , or, __VA_ARGS__ ) ) )
#define if_null( ... ) if( CHAIN(, is null, and, __VA_ARGS__ ) )
#define if_not_null( ... ) if( CHAIN(, isnt null, and, __VA_ARGS__ ) )
#define if_zero( ... ) if( CHAIN(, is 0, and, __VA_ARGS__ ) )
#define elif( ... ) else if( __VA_ARGS__ )

#define while_any( ... ) while( CHAIN(, , or, __VA_ARGS__ ) )
#define while_not( ... ) while( CHAIN( not, , and, __VA_ARGS__ ) )
#define while_null( ... ) while( CHAIN(, is null, and, __VA_ARGS__ ) )
#define while_not_null( ... ) while( CHAIN(, isnt null, and, __VA_ARGS__ ) )
#define while( ... ) while( CHAIN(, , and, __VA_ARGS__ ) )

#define leave_if( ... ) \
	if( __VA_ARGS__ ) leave
#define skip_if( ... ) \
	if( __VA_ARGS__ ) skip
#define out_if( ... ) \
	if( __VA_ARGS__ ) out

// // // // // // //
// scope management

#define _scope( COUNTER ) \
	for( register char JOIN( _scope_, COUNTER ) = 1; JOIN( _scope_, COUNTER )--; )
#define scope _scope( __COUNTER__ )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// structures

#define _struct_make( NAME, REF, ... ) \
	type_from( struct NAME REF ) NAME; \
	__VA_ARGS__ \
	struct PACKED NAME
#define struct( ... ) _struct_make( __VA_ARGS__, )

#define union( NAME ) \
	type_from( union NAME ) NAME; \
	union PACKED NAME
#define enum( NAME, TYPE ) \
	type_from( TYPE ) NAME; \
	enum PACKED NAME

#define make( STRUCT, ... ) ( (STRUCT){ __VA_ARGS__ } )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// variadic arguments

#ifndef va_start
#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST
type_from( __builtin_va_list ) __gnuc_va_list;
#endif
#define va_start( v, l ) __builtin_va_start( v, l )
#define va_end( v ) __builtin_va_end( v )
#define va_arg( v, l ) __builtin_va_arg( v, l )
#define va_copy( d, s ) __builtin_va_copy( d, s )
type_from( __gnuc_va_list ) va_list;
#endif

#define args_start( ARG_BEFORE_ELLIPSIS ) \
	va_list _args; \
	va_start( _args, ARG_BEFORE_ELLIPSIS )
#define args_reset( ARG_BEFORE_ELLIPSIS ) \
	va_end( _args ); \
	va_start( _args, ARG_BEFORE_ELLIPSIS )
#define args_next( TYPE ) _args_next_##TYPE

#define _args_next_byte va_arg( _args, unsigned int )
#define _args_next_u1 _args_next_byte
#define _args_next_s1 va_arg( _args, signed int )
#define _args_next_u2 va_arg( _args, unsigned int )
#define _args_next_s2 va_arg( _args, signed int )
#define _args_next_u4 va_arg( _args, unsigned int )
#define _args_next_s4 va_arg( _args, signed int )
#define _args_next_f4 va_arg( _args, double )
#define _args_next_u8 va_arg( _args, unsigned long long )
#define _args_next_s8 va_arg( _args, signed long long )
#define _args_next_f8 va_arg( _args, double )
#define _args_next_ref va_arg( _args, long ref )
#define args_next( TYPE ) _args_next_##TYPE

// // // // // // //
// bit operations

#define bit( N ) ( 1 << ( N - 1 ) )
#define bit_set( VAL, N ) ( ( VAL ) |= bit( N ) )
#define bit_unset( VAL, N ) ( ( VAL ) &= ~bit( N ) )
#define bit_flip( VAL, N ) ( ( VAL ) ^= bit( N ) )
#define get_bit( VAL, N ) ( ( VAL ) & bit( N ) )
#define make_bits( ... ) CHAIN(, , |, __VA_ARGS__ )

#define n_to_bits( N ) \
	pick( ( N ) <= 2, 1, \
		pick( ( N ) <= 4, 2, \
			pick( ( N ) <= 8, 3, \
				pick( ( N ) <= 16, 4, \
					pick( ( N ) <= 32, 5, \
						pick( ( N ) <= 64, 6, \
							pick( ( N ) <= 128, 7, pick( ( N ) <= 256, 8, 9 ) ) ) ) ) ) ) )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// types

// // // // // // //
// comparison and selection macros

#define MIN( A, B ) pick( ( A ) < ( B ), A, B )
#define MIN3( A, B, C ) MIN( A, MIN( B, C ) )
#define MIN4( A, B, C, D ) MIN( A, MIN3( B, C, D ) )
#define MIN5( A, B, C, D, E ) MIN( A, MIN4( B, C, D, E ) )
#define MAX( A, B ) pick( ( A ) > ( B ), A, B )
#define MAX3( A, B, C ) MAX( A, MAX( B, C ) )
#define MAX4( A, B, C, D ) MAX( A, MAX3( B, C, D ) )
#define MAX5( A, B, C, D, E ) MAX( A, MAX4( B, C, D, E ) )

// // // // // // //
// statistical operations

#define MEDIAN( A, B, C ) \
	pick( ( A ) > ( B ), pick( ( B ) > ( C ), B, MIN( A, C ) ), \
		pick( ( A ) > ( C ), A, MIN( B, C ) ) )
#define MEDIAN4( A, B, C, D ) \
	( ( MIN( MAX( A, B ), MAX( C, D ) ) + MAX( MIN( A, B ), MIN( C, D ) ) ) / 2 )
#define MEDIAN4_BITWISE( A, B, C, D ) \
	( ( MIN( MAX( A, B ), MAX( C, D ) ) + MAX( MIN( A, B ), MIN( C, D ) ) ) >> 1 )
#define MEDIAN5( A, B, C, D, E ) \
	( ( A + B + C + D + E - MIN5( A, B, C, D, E ) - MAX5( A, B, C, D, E ) ) / 3 )

#define AVG( A, B ) ( ( ( A ) + ( B ) ) / 2 )
#define AVG_BITWISE( A, B ) ( ( ( A ) + ( B ) ) >> 1 )
#define AVG3( A, B, C ) ( ( ( A ) + ( B ) + ( C ) ) / 3 )
#define AVG4( A, B, C, D ) ( ( ( A ) + ( B ) + ( C ) + ( D ) ) / 4 )
#define AVG4_BITWISE( A, B, C, D ) ( ( ( A ) + ( B ) + ( C ) + ( D ) ) >> 2 )

// // // // // // //

#define CLAMP( V, MINIMUM, MAXIMUM ) \
	MIN( MAX( ( V ), ( MINIMUM ) ), ( MAXIMUM ) )
#define SATURATE( V ) CLAMP( V, 0, 1 )
#define SQR( V ) ( ( V ) * ( V ) )
#define CUBE( V ) ( ( V ) * ( V ) * ( V ) )
#define ABS( V ) pick( ( V ) < 0, -( V ), V )
#define SIGN( V ) pick( ( V ) >= 0, 1, -1 )
#define SIGN_ZERO( V ) pick( ( V ) > 0, 1, pick( ( V ) < 0, -1, 0 ) )

// // // // // // //
// interpolation and snapping

#define MIX( A, B, AMOUNT ) ( ( A ) + ( ( AMOUNT ) * ( B - A ) ) )
#define MAP( V, A_MIN, A_MAX, B_MIN, B_MAX ) \
	( ( B_MIN ) + ( ( V - A_MIN ) * ( ( B_MAX - B_MIN ) / ( A_MAX - A_MIN ) ) ) )
#define RANGE( V, LOWER, UPPER ) ( ( V - LOWER ) / ( UPPER - LOWER ) )

#define SNAP( V, MULTIPLES_OF ) ( ( V ) / ( MULTIPLES_OF ) ) * ( MULTIPLES_OF )
#define SNAP_f( N, V, MULTIPLES_OF ) \
	f##N##_trunc( ( V ) / ( MULTIPLES_OF ) ) * ( MULTIPLES_OF )
#define SNAP_BIT( V, BIT ) ( ( ( V ) >> ( BIT ) ) << ( BIT ) )

// // // // // // //
// float-specific operations

#define TRUNC_f( N, ... ) f##N( s##N( __VA_ARGS__ ) )
#define FLOOR_f( N, ... ) \
	pick( __VA_ARGS__ >= 0, TRUNC_f( N, __VA_ARGS__ ), \
		f##N( s##N( __VA_ARGS__ ) - 1 ) )
#define ROUND_f( N, ... ) \
	TRUNC_f( N, __VA_ARGS__ + pick( __VA_ARGS__ >= 0, .5, -.5 ) )
#define CEIL_f( N, ... ) \
	pick( __VA_ARGS__ > 0, f##N( s##N( __VA_ARGS__ ) + 1 ), \
		TRUNC_f( N, __VA_ARGS__ ) )
#define MOD_f( N, V, MODULO ) \
	( ( V ) - ( MODULO ) * f##N##_floor( ( V ) / ( MODULO ) ) )

#define SMOOTH( V ) \
	pick( ( V ) <= 0., 0., \
		pick( ( V ) >= 1., 1., ( V ) * ( V ) * ( 3. - 2. * ( V ) ) ) )

// // // // // // //
// function group generators

#define FUNCTION_GROUP_BASE( T, N ) \
	embed T##N T##N##_min( in T##N a, in T##N b ) { out MIN( a, b ); } \
	embed T##N T##N##_min3( in T##N a, in T##N b, in T##N c ) \
	{ \
		out MIN3( a, b, c ); \
	} \
	embed T##N T##N##_min4( in T##N a, in T##N b, in T##N c, in T##N d ) \
	{ \
		out MIN4( a, b, c, d ); \
	} \
	embed T##N T##N##_min5( \
		in T##N a, in T##N b, in T##N c, in T##N d, in T##N e ) \
	{ \
		out MIN5( a, b, c, d, e ); \
	} \
	embed T##N T##N##_max( in T##N a, in T##N b ) { out MAX( a, b ); } \
	embed T##N T##N##_max3( in T##N a, in T##N b, in T##N c ) \
	{ \
		out MAX3( a, b, c ); \
	} \
	embed T##N T##N##_max4( in T##N a, in T##N b, in T##N c, in T##N d ) \
	{ \
		out MAX4( a, b, c, d ); \
	} \
	embed T##N T##N##_max5( \
		in T##N a, in T##N b, in T##N c, in T##N d, in T##N e ) \
	{ \
		out MAX5( a, b, c, d, e ); \
	} \
	embed T##N T##N##_median( in T##N a, in T##N b, in T##N c ) \
	{ \
		out MEDIAN( a, b, c ); \
	} \
	embed T##N T##N##_median5( \
		in T##N a, in T##N b, in T##N c, in T##N d, in T##N e ) \
	{ \
		out MEDIAN5( a, b, c, d, e ); \
	} \
	embed T##N T##N##_avg3( in T##N a, in T##N b, in T##N c ) \
	{ \
		out AVG3( a, b, c ); \
	} \
	embed T##N T##N##_clamp( in T##N v, in T##N min, in T##N max ) \
	{ \
		out CLAMP( v, min, max ); \
	} \
	embed T##N T##N##_saturate( in T##N v ) { out SATURATE( v ); } \
	embed T##N T##N##_sqr( in T##N v ) { out SQR( v ); } \
	embed T##N T##N##_cube( in T##N v ) { out CUBE( v ); }

#define FUNCTION_GROUP_BASE_US( US, N ) \
	embed US##N US##N##_median4( \
		in US##N a, in US##N b, in US##N c, in US##N d ) \
	{ \
		out MEDIAN4_BITWISE( a, b, c, d ); \
	} \
	embed US##N US##N##_avg( in US##N a, in US##N b ) \
	{ \
		out AVG_BITWISE( a, b ); \
	} \
	embed US##N US##N##_avg4( in US##N a, in US##N b, in US##N c, in US##N d ) \
	{ \
		out AVG4_BITWISE( a, b, c, d ); \
	} \
	embed US##N US##N##_snap( in US##N v, in US##N multiples_of ) \
	{ \
		out SNAP( v, multiples_of ); \
	} \
	embed US##N US##N##_snap_bit( in US##N v, in US##N b ) \
	{ \
		out SNAP_BIT( v, b ); \
	}

#define FUNCTION_GROUP_U( N ) \
	FUNCTION_GROUP_BASE( u, N ) \
	FUNCTION_GROUP_BASE_US( u, N )

#define FUNCTION_GROUP_S( N ) \
	FUNCTION_GROUP_BASE( s, N ) \
	FUNCTION_GROUP_BASE_US( s, N ) \
	embed s##N s##N##_abs( in s##N v ) { out ABS( v ); } \
	embed s##N s##N##_sign( in s##N v ) { out SIGN( v ); } \
	embed s##N s##N##_sign_zero( in s##N v ) { out SIGN_ZERO( v ); }

#define FUNCTION_GROUP_F( N ) \
	FUNCTION_GROUP_BASE( f, N ) \
	embed f##N f##N##_trunc( in f##N v ) { out TRUNC_f( N, v ); } \
	embed f##N f##N##_floor( in f##N v ) { out FLOOR_f( N, v ); } \
	embed f##N f##N##_round( in f##N v ) { out ROUND_f( N, v ); } \
	embed f##N f##N##_ceil( in f##N v ) { out CEIL_f( N, v ); } \
	embed f##N f##N##_mod( in f##N v, in f##N m ) { out MOD_f( N, v, m ); } \
	embed f##N f##N##_median4( in f##N a, in f##N b, in f##N c, in f##N d ) \
	{ \
		out MEDIAN4( a, b, c, d ); \
	} \
	embed f##N f##N##_avg( in f##N a, in f##N b ) { out AVG( a, b ); } \
	embed f##N f##N##_avg4( in f##N a, in f##N b, in f##N c, in f##N d ) \
	{ \
		out AVG4( a, b, c, d ); \
	} \
	embed f##N f##N##_abs( in f##N v ) { out ABS( v ); } \
	embed f##N f##N##_sign( in f##N v ) { out SIGN( v ); } \
	embed f##N f##N##_sign_zero( in f##N v ) { out SIGN_ZERO( v ); } \
	embed f##N f##N##_snap( in f##N v, in f##N multiples_of ) \
	{ \
		out SNAP_f( N, v, multiples_of ); \
	} \
	embed f##N f##N##_mix( in f##N a, in f##N b, in f##N amount ) \
	{ \
		out MIX( a, b, amount ); \
	} \
	embed f##N f##N##_map( in f##N v, in f##N from_lower, in f##N from_upper, \
		in f##N to_lower, in f##N to_upper ) \
	{ \
		out MAP( v, from_lower, from_upper, to_lower, to_upper ); \
	} \
	embed f##N f##N##_range( in f##N v, in f##N lower, in f##N upper ) \
	{ \
		out RANGE( v, lower, upper ); \
	} \
	embed f##N f##N##_smooth( in f##N v ) { out SMOOTH( v ); }

// // // // // // //
// vector construction and functions

#define VECTOR_TYPES( TYPE ) \
	struct( TYPE##_2d ) \
	{ \
		union \
		{ \
			TYPE x; \
			TYPE w; \
		}; \
		union \
		{ \
			TYPE y; \
			TYPE h; \
		}; \
	}; \
	struct( TYPE##_3d ) \
	{ \
		union \
		{ \
			TYPE x; \
			TYPE w; \
			TYPE r; \
		}; \
		union \
		{ \
			TYPE y; \
			TYPE h; \
			TYPE g; \
		}; \
		union \
		{ \
			TYPE z; \
			TYPE d; \
			TYPE b; \
		}; \
	}; \
	struct( TYPE##_4d ) \
	{ \
		union \
		{ \
			TYPE x; \
			TYPE r; \
		}; \
		union \
		{ \
			TYPE y; \
			TYPE g; \
		}; \
		union \
		{ \
			TYPE z; \
			TYPE b; \
		}; \
		union \
		{ \
			TYPE w; \
			TYPE a; \
		}; \
	};

#define VECTOR_FUNCTIONS_BASE( TYPE ) \
	embed TYPE##_2d TYPE##_2d_add( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( a.x + b.x, a.y + b.y ); \
	} \
	embed TYPE##_2d TYPE##_2d_sub( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( a.x - b.x, a.y - b.y ); \
	} \
	embed TYPE##_2d TYPE##_2d_mul( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( a.x * b.x, a.y * b.y ); \
	} \
	embed TYPE##_2d TYPE##_2d_div( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( a.x / b.x, a.y / b.y ); \
	} \
	embed TYPE TYPE##_2d_dot( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out a.x* b.x + a.y* b.y; \
	} \
	embed TYPE TYPE##_2d_mag( in TYPE##_2d a ) { out a.x* a.x + a.y* a.y; } \
	embed TYPE##_2d TYPE##_2d_add_##TYPE( in TYPE##_2d a, in TYPE v ) \
	{ \
		out make_##TYPE##_2d( a.x + v, a.y + v ); \
	} \
	embed TYPE##_2d TYPE##_2d_sub_##TYPE( in TYPE##_2d a, in TYPE v ) \
	{ \
		out make_##TYPE##_2d( a.x - v, a.y - v ); \
	} \
	embed TYPE##_2d TYPE##_2d_mul_##TYPE( in TYPE##_2d a, in TYPE v ) \
	{ \
		out make_##TYPE##_2d( a.x * v, a.y * v ); \
	} \
	embed TYPE##_2d TYPE##_2d_div_##TYPE( in TYPE##_2d a, in TYPE v ) \
	{ \
		out make_##TYPE##_2d( a.x / v, a.y / v ); \
	} \
	embed TYPE##_3d TYPE##_3d_add( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( a.x + b.x, a.y + b.y, a.z + b.z ); \
	} \
	embed TYPE##_3d TYPE##_3d_sub( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( a.x - b.x, a.y - b.y, a.z - b.z ); \
	} \
	embed TYPE##_3d TYPE##_3d_mul( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( a.x * b.x, a.y * b.y, a.z * b.z ); \
	} \
	embed TYPE##_3d TYPE##_3d_div( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( a.x / b.x, a.y / b.y, a.z / b.z ); \
	} \
	embed TYPE TYPE##_3d_dot( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out a.x* b.x + a.y* b.y + a.z* b.z; \
	} \
	embed TYPE TYPE##_3d_mag( in TYPE##_3d a ) \
	{ \
		out a.x* a.x + a.y* a.y + a.z* a.z; \
	} \
	embed TYPE##_3d TYPE##_3d_add_##TYPE( in TYPE##_3d a, in TYPE v ) \
	{ \
		out make_##TYPE##_3d( a.x + v, a.y + v, a.z + v ); \
	} \
	embed TYPE##_3d TYPE##_3d_sub_##TYPE( in TYPE##_3d a, in TYPE v ) \
	{ \
		out make_##TYPE##_3d( a.x - v, a.y - v, a.z - v ); \
	} \
	embed TYPE##_3d TYPE##_3d_mul_##TYPE( in TYPE##_3d a, in TYPE v ) \
	{ \
		out make_##TYPE##_3d( a.x * v, a.y * v, a.z * v ); \
	} \
	embed TYPE##_3d TYPE##_3d_div_##TYPE( in TYPE##_3d a, in TYPE v ) \
	{ \
		out make_##TYPE##_3d( a.x / v, a.y / v, a.z / v ); \
	} \
	embed TYPE##_4d TYPE##_4d_add( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); \
	} \
	embed TYPE##_4d TYPE##_4d_sub( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); \
	} \
	embed TYPE##_4d TYPE##_4d_mul( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); \
	} \
	embed TYPE##_4d TYPE##_4d_div( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w ); \
	} \
	embed TYPE TYPE##_4d_dot( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out a.x* b.x + a.y* b.y + a.z* b.z + a.w* b.w; \
	} \
	embed TYPE TYPE##_4d_mag( in TYPE##_4d a ) \
	{ \
		out a.x* a.x + a.y* a.y + a.z* a.z + a.w* a.w; \
	} \
	embed TYPE##_4d TYPE##_4d_add_##TYPE( in TYPE##_4d a, in TYPE v ) \
	{ \
		out make_##TYPE##_4d( a.x + v, a.y + v, a.z + v, a.w + v ); \
	} \
	embed TYPE##_4d TYPE##_4d_sub_##TYPE( in TYPE##_4d a, in TYPE v ) \
	{ \
		out make_##TYPE##_4d( a.x - v, a.y - v, a.z - v, a.w - v ); \
	} \
	embed TYPE##_4d TYPE##_4d_mul_##TYPE( in TYPE##_4d a, in TYPE v ) \
	{ \
		out make_##TYPE##_4d( a.x * v, a.y * v, a.z * v, a.w * v ); \
	} \
	embed TYPE##_4d TYPE##_4d_div_##TYPE( in TYPE##_4d a, in TYPE v ) \
	{ \
		out make_##TYPE##_4d( a.x / v, a.y / v, a.z / v, a.w / v ); \
	}

#define VECTOR_FUNCTIONS_U( TYPE ) \
	VECTOR_FUNCTIONS_BASE( TYPE ) \
	embed TYPE##_2d TYPE##_2d_min( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( TYPE##_min( a.x, b.x ), TYPE##_min( a.y, b.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_max( in TYPE##_2d a, in TYPE##_2d b ) \
	{ \
		out make_##TYPE##_2d( TYPE##_max( a.x, b.x ), TYPE##_max( a.y, b.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_clamp( \
		in TYPE##_2d v, in TYPE##_2d min, in TYPE##_2d max ) \
	{ \
		out make_##TYPE##_2d( \
			TYPE##_clamp( v.x, min.x, max.x ), TYPE##_clamp( v.y, min.y, max.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_saturate( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_saturate( v.x ), TYPE##_saturate( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_sqr( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_sqr( v.x ), TYPE##_sqr( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_cube( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_cube( v.x ), TYPE##_cube( v.y ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_min( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( TYPE##_min( a.x, b.x ), TYPE##_min( a.y, b.y ), \
			TYPE##_min( a.z, b.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_max( in TYPE##_3d a, in TYPE##_3d b ) \
	{ \
		out make_##TYPE##_3d( TYPE##_max( a.x, b.x ), TYPE##_max( a.y, b.y ), \
			TYPE##_max( a.z, b.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_clamp( \
		in TYPE##_3d v, in TYPE##_3d min, in TYPE##_3d max ) \
	{ \
		out make_##TYPE##_3d( TYPE##_clamp( v.x, min.x, max.x ), \
			TYPE##_clamp( v.y, min.y, max.y ), TYPE##_clamp( v.z, min.z, max.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_saturate( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( TYPE##_saturate( v.x ), TYPE##_saturate( v.y ), \
			TYPE##_saturate( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_sqr( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_sqr( v.x ), TYPE##_sqr( v.y ), TYPE##_sqr( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_cube( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_cube( v.x ), TYPE##_cube( v.y ), TYPE##_cube( v.z ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_min( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( TYPE##_min( a.x, b.x ), TYPE##_min( a.y, b.y ), \
			TYPE##_min( a.z, b.z ), TYPE##_min( a.w, b.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_max( in TYPE##_4d a, in TYPE##_4d b ) \
	{ \
		out make_##TYPE##_4d( TYPE##_max( a.x, b.x ), TYPE##_max( a.y, b.y ), \
			TYPE##_max( a.z, b.z ), TYPE##_max( a.w, b.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_clamp( \
		in TYPE##_4d v, in TYPE##_4d min, in TYPE##_4d max ) \
	{ \
		out make_##TYPE##_4d( TYPE##_clamp( v.x, min.x, max.x ), \
			TYPE##_clamp( v.y, min.y, max.y ), TYPE##_clamp( v.z, min.z, max.z ), \
			TYPE##_clamp( v.w, min.w, max.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_saturate( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_saturate( v.x ), TYPE##_saturate( v.y ), \
			TYPE##_saturate( v.z ), TYPE##_saturate( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_sqr( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_sqr( v.x ), TYPE##_sqr( v.y ), \
			TYPE##_sqr( v.z ), TYPE##_sqr( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_cube( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_cube( v.x ), TYPE##_cube( v.y ), \
			TYPE##_cube( v.z ), TYPE##_cube( v.w ) ); \
	}

#define VECTOR_FUNCTIONS_S( TYPE ) \
	VECTOR_FUNCTIONS_U( TYPE ) \
	embed TYPE##_2d TYPE##_2d_abs( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_abs( v.x ), TYPE##_abs( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_sign( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_sign( v.x ), TYPE##_sign( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_sign_zero( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_sign_zero( v.x ), TYPE##_sign_zero( v.y ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_abs( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_abs( v.x ), TYPE##_abs( v.y ), TYPE##_abs( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_sign( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_sign( v.x ), TYPE##_sign( v.y ), TYPE##_sign( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_sign_zero( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( TYPE##_sign_zero( v.x ), TYPE##_sign_zero( v.y ), \
			TYPE##_sign_zero( v.z ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_abs( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_abs( v.x ), TYPE##_abs( v.y ), \
			TYPE##_abs( v.z ), TYPE##_abs( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_sign( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_sign( v.x ), TYPE##_sign( v.y ), \
			TYPE##_sign( v.z ), TYPE##_sign( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_sign_zero( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_sign_zero( v.x ), TYPE##_sign_zero( v.y ), \
			TYPE##_sign_zero( v.z ), TYPE##_sign_zero( v.w ) ); \
	}

#define VECTOR_FUNCTIONS_F( TYPE ) \
	VECTOR_FUNCTIONS_S( TYPE ) \
	embed TYPE##_2d TYPE##_2d_trunc( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_trunc( v.x ), TYPE##_trunc( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_floor( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_floor( v.x ), TYPE##_floor( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_round( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_round( v.x ), TYPE##_round( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_ceil( in TYPE##_2d v ) \
	{ \
		out make_##TYPE##_2d( TYPE##_ceil( v.x ), TYPE##_ceil( v.y ) ); \
	} \
	embed TYPE##_2d TYPE##_2d_mix( \
		in TYPE##_2d a, in TYPE##_2d b, in TYPE##_2d amount ) \
	{ \
		out make_##TYPE##_2d( \
			TYPE##_mix( a.x, b.x, amount.x ), TYPE##_mix( a.y, b.y, amount.y ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_trunc( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_trunc( v.x ), TYPE##_trunc( v.y ), TYPE##_trunc( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_floor( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_floor( v.x ), TYPE##_floor( v.y ), TYPE##_floor( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_round( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_round( v.x ), TYPE##_round( v.y ), TYPE##_round( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_ceil( in TYPE##_3d v ) \
	{ \
		out make_##TYPE##_3d( \
			TYPE##_ceil( v.x ), TYPE##_ceil( v.y ), TYPE##_ceil( v.z ) ); \
	} \
	embed TYPE##_3d TYPE##_3d_mix( \
		in TYPE##_3d a, in TYPE##_3d b, in TYPE##_3d amount ) \
	{ \
		out make_##TYPE##_3d( TYPE##_mix( a.x, b.x, amount.x ), \
			TYPE##_mix( a.y, b.y, amount.y ), TYPE##_mix( a.z, b.z, amount.z ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_trunc( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_trunc( v.x ), TYPE##_trunc( v.y ), \
			TYPE##_trunc( v.z ), TYPE##_trunc( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_floor( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_floor( v.x ), TYPE##_floor( v.y ), \
			TYPE##_floor( v.z ), TYPE##_floor( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_round( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_round( v.x ), TYPE##_round( v.y ), \
			TYPE##_round( v.z ), TYPE##_round( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_ceil( in TYPE##_4d v ) \
	{ \
		out make_##TYPE##_4d( TYPE##_ceil( v.x ), TYPE##_ceil( v.y ), \
			TYPE##_ceil( v.z ), TYPE##_ceil( v.w ) ); \
	} \
	embed TYPE##_4d TYPE##_4d_mix( \
		in TYPE##_4d a, in TYPE##_4d b, in TYPE##_4d amount ) \
	{ \
		out make_##TYPE##_4d( TYPE##_mix( a.x, b.x, amount.x ), \
			TYPE##_mix( a.y, b.y, amount.y ), TYPE##_mix( a.z, b.z, amount.z ), \
			TYPE##_mix( a.w, b.w, amount.w ) ); \
	}

// // // // // // //
// 1 byte

type_from( unsigned char ) byte;
#define byte( ... ) to( byte, __VA_ARGS__ )

type_from( byte ) u1;
#define u1( ... ) to( u1, __VA_ARGS__ )
#define min_u1 u1( 0x00u )
#define max_u1 u1( 0xffu )
FUNCTION_GROUP_U( 1 )
VECTOR_TYPES( u1 )
#define u1_2d( ... ) { DEFAULTS( ( 0, 0 ), __VA_ARGS__ ) }
#define make_u1_2d( ... ) ( u1_2d ) u1_2d( __VA_ARGS__ )
#define to_u1_2d( _2d ) make_u1_2d( u1( _2d.x ), u1( _2d.y ) )
#define u1_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_u1_3d( ... ) ( u1_3d ) u1_3d( __VA_ARGS__ )
#define to_u1_3d( _3d ) make_u1_3d( u1( _3d.x ), u1( _3d.y ), u1( _3d.z ) )
#define u1_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_u1_4d( ... ) ( u1_4d ) u1_4d( __VA_ARGS__ )
#define to_u1_4d( _4d ) \
	make_u1_4d( u1( _4d.x ), u1( _4d.y ), u1( _4d.z ), u1( _4d.w )
VECTOR_FUNCTIONS_U( u1 )

type_from( signed char ) s1;
#define s1( ... ) to( s1, __VA_ARGS__ )
#define min_s1 s1( 0x80 )
#define max_s1 s1( 0x7F )
FUNCTION_GROUP_S( 1 )
VECTOR_TYPES( s1 )
#define s1_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_s1_2d( ... ) ( s1_2d ) s1_2d( __VA_ARGS__ )
#define to_s1_2d( _2d ) make_s1_2d( s1( _2d.x ), s1( _2d.y ) )
#define s1_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_s1_3d( ... ) ( s1_3d ) s1_3d( __VA_ARGS__ )
#define to_s1_3d( _3d ) make_s1_3d( s1( _3d.x ), s1( _3d.y ), s1( _3d.z ) )
#define s1_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_s1_4d( ... ) ( s1_4d ) s1_4d( __VA_ARGS__ )
#define to_s1_4d( _4d ) \
	make_s1_4d( s1( _4d.x ), s1( _4d.y ), s1( _4d.z ), s1( _4d.w ) )
VECTOR_FUNCTIONS_S( s1 )

// // // // // // //
// 2 bytes

type_from( unsigned short ) u2;
#define u2( ... ) to( u2, __VA_ARGS__ )
#define min_u2 u2( 0x0000u )
#define max_u2 u2( 0xFFFFu )
FUNCTION_GROUP_U( 2 )
VECTOR_TYPES( u2 )
#define u2_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_u2_2d( ... ) ( u2_2d ) u2_2d( __VA_ARGS__ )
#define to_u2_2d( _2d ) make_u2_2d( u2( _2d.x ), u2( _2d.y ) )
#define u2_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_u2_3d( ... ) ( u2_3d ) u2_3d( __VA_ARGS__ )
#define to_u2_3d( _3d ) make_u2_3d( u2( _3d.x ), u2( _3d.y ), u2( _3d.z ) )
#define u2_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_u2_4d( ... ) ( u2_4d ) u2_4d( __VA_ARGS__ )
#define to_u2_4d( _4d ) \
	make_u2_4d( u2( _4d.x ), u2( _4d.y ), u2( _4d.z ), u2( _4d.w ) )
VECTOR_FUNCTIONS_U( u2 )

type_from( signed short ) s2;
#define s2( ... ) to( s2, __VA_ARGS__ )
#define min_s2 s2( 0x8000 )
#define max_s2 s2( 0x7FFF )
FUNCTION_GROUP_S( 2 )
VECTOR_TYPES( s2 )
#define s2_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_s2_2d( ... ) ( s2_2d ) s2_2d( __VA_ARGS__ )
#define to_s2_2d( _2d ) make_s2_2d( s2( _2d.x ), s2( _2d.y ) )
#define s2_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_s2_3d( ... ) ( s2_3d ) s2_3d( __VA_ARGS__ )
#define to_s2_3d( _3d ) make_s2_3d( s2( _3d.x ), s2( _3d.y ), s2( _3d.z ) )
#define s2_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_s2_4d( ... ) ( s2_4d ) s2_4d( __VA_ARGS__ )
#define to_s2_4d( _4d ) \
	make_s2_4d( s2( _4d.x ), s2( _4d.y ), s2( _4d.z ), s2( _4d.w ) )
VECTOR_FUNCTIONS_S( s2 )

// // // // // // //
// 4 bytes

type_from( unsigned int ) u4;
#define u4( ... ) to( u4, __VA_ARGS__ )
#define min_u4 u4( 0x00000000u )
#define max_u4 u4( 0xFFFFFFFFu )
FUNCTION_GROUP_U( 4 )
VECTOR_TYPES( u4 )
#define u4_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_u4_2d( ... ) ( u4_2d ) u4_2d( __VA_ARGS__ )
#define to_u4_2d( _2d ) make_u4_2d( u4( _2d.x ), u4( _2d.y ) )
#define u4_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_u4_3d( ... ) ( u4_3d ) u4_3d( __VA_ARGS__ )
#define to_u4_3d( _3d ) make_u4_3d( u4( _3d.x ), u4( _3d.y ), u4( _3d.z ) )
#define u4_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_u4_4d( ... ) ( u4_4d ) u4_4d( __VA_ARGS__ )
#define to_u4_4d( _4d ) \
	make_u4_4d( u4( _4d.x ), u4( _4d.y ), u4( _4d.z ), u4( _4d.w ) )
VECTOR_FUNCTIONS_U( u4 )

type_from( signed int ) s4;
#define s4( ... ) to( s4, __VA_ARGS__ )
#define min_s4 s4( 0x80000000 )
#define max_s4 s4( 0x7FFFFFFF )
FUNCTION_GROUP_S( 4 )
VECTOR_TYPES( s4 )
#define s4_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_s4_2d( ... ) ( s4_2d ) s4_2d( __VA_ARGS__ )
#define to_s4_2d( _2d ) make_s4_2d( s4( _2d.x ), s4( _2d.y ) )
#define s4_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_s4_3d( ... ) ( s4_3d ) s4_3d( __VA_ARGS__ )
#define to_s4_3d( _3d ) make_s4_3d( s4( _3d.x ), s4( _4d.y ), s4( _3d.z ) )
#define s4_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_s4_4d( ... ) ( s4_4d ) s4_4d( __VA_ARGS__ )
#define to_s4_4d( _4d ) \
	make_s4_4d( s4( _4d.x ), s4( _4d.y ), s4( _4d.z ), s4( _4d.w ) )
VECTOR_FUNCTIONS_S( s4 )

type_from( float ) f4;
#define f4( ... ) to( f4, __VA_ARGS__ )
FUNCTION_GROUP_F( 4 )
VECTOR_TYPES( f4 )
#define f4_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_f4_2d( ... ) ( f4_2d ) f4_2d( __VA_ARGS__ )
#define to_f4_2d( _2d ) make_f4_2d( f4( _2d.x ), f4( _2d.y ) )
#define f4_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_f4_3d( ... ) ( f4_3d ) f4_3d( __VA_ARGS__ )
#define to_f4_3d( _3d ) make_f4_3d( f4( _3d.x ), f4( _3d.y ), f4( _3d.z ) )
#define f4_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_f4_4d( ... ) ( f4_4d ) f4_4d( __VA_ARGS__ )
#define to_f4_4d( _4d ) \
	make_f4_4d( f4( _4d.x ), f4( _4d.y ), f4( _4d.z ), f4( _4d.w ) )
VECTOR_FUNCTIONS_F( f4 )

// // // // // // //
// 8 bytes

type_from( unsigned long long ) u8;
#define u8( ... ) to( u8, __VA_ARGS__ )
#define min_u8 u8( 0x0000000000000000u )
#define max_u8 u8( 0xFFFFFFFFFFFFFFFFu )
FUNCTION_GROUP_U( 8 )
VECTOR_TYPES( u8 )
#define u8_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_u8_2d( ... ) ( u8_2d ) u8_2d( __VA_ARGS__ )
#define to_u8_2d( _2d ) make_u8_2d( u8( _2d.x ), u8( _2d.y ) )
#define u8_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_u8_3d( ... ) ( u8_3d ) u8_3d( __VA_ARGS__ )
#define to_u8_3d( _3d ) make_u8_3d( u8( _3d.x ), u8( _3d.y ), u8( _3d.z ) )
#define u8_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_u8_4d( ... ) ( u8_4d ) u8_4d( __VA_ARGS__ )
#define to_u8_4d( _4d ) \
	make_u8_4d( u8( _4d.x ), u8( _4d.y ), u8( _4d.z ), u8( _4d.w ) )
VECTOR_FUNCTIONS_U( u8 )

type_from( signed long long ) s8;
#define s8( ... ) to( s8, __VA_ARGS__ )
#define min_s8 s8( 0x8000000000000000 )
#define max_s8 s8( 0x7FFFFFFFFFFFFFFF )
FUNCTION_GROUP_S( 8 )
VECTOR_TYPES( s8 )
#define s8_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_s8_2d( ... ) ( s8_2d ) s8_2d( __VA_ARGS__ )
#define to_s8_2d( _2d ) make_s8_2d( s8( _2d.x ), s8( _2d.y ) )
#define s8_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_s8_3d( ... ) ( s8_3d ) s8_3d( __VA_ARGS__ )
#define to_s8_3d( _3d ) make_s8_3d( s8( _3d.x ), s8( _3d.y ), s8( _3d.z ) )
#define s8_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_s8_4d( ... ) ( s8_4d ) s8_4d( __VA_ARGS__ )
#define to_s8_4d( _4d ) \
	make_s8_4d( s8( _4d.x ), s8( _4d.y ), s8( _4d.z ), s8( _4d.w ) )
VECTOR_FUNCTIONS_S( s8 )

type_from( double ) f8;
#define f8( ... ) to( f8, __VA_ARGS__ )
FUNCTION_GROUP_F( 8 )
VECTOR_TYPES( f8 )
#define f8_2d( ... ) CONSTRUCT( ( x, y ), ( 0, 0 ), __VA_ARGS__ )
#define make_f8_2d( ... ) ( f8_2d ) f8_2d( __VA_ARGS__ )
#define to_f8_2d( _2d ) make_f8_2d( f8( _2d.x ), f8( _2d.y ) )
#define f8_3d( ... ) CONSTRUCT( ( x, y, z ), ( 0, 0, 0 ), __VA_ARGS__ )
#define make_f8_3d( ... ) ( f8_3d ) f8_3d( __VA_ARGS__ )
#define to_f8_3d( _3d ) make_f8_3d( f8( _3d.x ), f8( _3d.y ), f8( _3d.z ) )
#define f8_4d( ... ) CONSTRUCT( ( x, y, z, w ), ( 0, 0, 0, 0 ), __VA_ARGS__ )
#define make_f8_4d( ... ) ( f8_4d ) f8_4d( __VA_ARGS__ )
#define to_f8_4d( _4d ) \
	make_f8_4d( f8( _4d.x ), f8( _4d.y ), f8( _4d.z ), f8( _4d.w ) )
VECTOR_FUNCTIONS_F( f8 )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
///

#define _range( POS_NAME, FROM, TO, SYMBOL, COUNTER ) \
	for( temp s4 POS_NAME = ( FROM ); POS_NAME SYMBOL( s4( TO ) ); ++POS_NAME )
#define range( POS_NAME, FROM, TO ) \
	_range( POS_NAME, FROM, TO, <=, __COUNTER__ )

#define _range_step( POS_NAME, FROM, TO, STEP, SYMBOL, COUNTER ) \
	for( temp s4 POS_NAME = ( FROM ); POS_NAME SYMBOL( TO ); POS_NAME += STEP )
#define range_step( POS_NAME, FROM, TO, STEP ) \
	_range_step( POS_NAME, FROM, TO, STEP, <=, __COUNTER__ )

#define iter_step( POS_NAME, SIZE, STEP ) \
	_range_step( POS_NAME, 0, SIZE, STEP, <, __COUNTER__ )
#define iter( POS_NAME, SIZE ) _range( POS_NAME, 0, SIZE, <, __COUNTER__ )

#define iter_grid( X_NAME, Y_NAME, WIDTH, HEIGHT ) \
	iter( Y_NAME, HEIGHT ) for( temp s4 X_NAME = 0; X_NAME < WIDTH; ++X_NAME )

#define _repeat( N_TIMES, COUNTER ) iter( JOIN( _REP_, COUNTER ), N_TIMES )
#define repeat( N_TIMES ) _repeat( s4( N_TIMES ), __COUNTER__ )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// byte operations

embed byte ref copy_bytes( mutable byte mutable_ref from_ref, mutable u8 size,
	mutable byte mutable_ref to_ref )
{
	temp const u8 mutable_ref from_8 = to( u8 mutable_ref, from_ref );
	temp u8 mutable_ref to_8 = to( u8 mutable_ref, to_ref );
	//
	while( size >= 64 )
	{
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		val_of( to_8++ ) = val_of( from_8++ );
		size -= 64;
	}
	//
	while( size >= 8 )
	{
		val_of( to_8++ ) = val_of( from_8++ );
		size -= 8;
	}
	//
	from_ref = to( byte mutable_ref, from_8 );
	to_ref = to( byte mutable_ref, to_8 );
	while( size-- ) { val_of( to_ref++ ) = val_of( from_ref++ ); }
	//
	out to( byte ref, to_ref );
}

embed byte ref move_bytes( mutable byte mutable_ref from_ref, mutable u8 size,
	mutable byte mutable_ref to_ref )
{
	out_if( to_ref < from_ref ) copy_bytes( from_ref, size, to_ref );
	//
	temp const u8 mutable_ref from_8 = to( u8 mutable_ref, from_ref + size );
	temp u8 mutable_ref to_8 = to( u8 mutable_ref, to_ref + size );
	//
	while( size >= 64 )
	{
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		val_of( --to_8 ) = val_of( --from_8 );
		size -= 64;
	}
	//
	while( size >= 8 )
	{
		val_of( --to_8 ) = val_of( --from_8 );
		size -= 8;
	}
	//
	from_ref = to( byte mutable_ref, from_8 );
	to_ref = to( byte mutable_ref, to_8 );
	while( size-- ) { val_of( --to_ref ) = val_of( --from_ref ); }
	//
	out to( byte ref, to_ref );
}

embed byte ref fill_bytes(
	mutable byte mutable_ref bytes_ref, in byte val, mutable u8 size )
{
	temp u8 pattern = val;
	pattern = ( pattern << 8 ) | pattern;
	pattern = ( pattern << 16 ) | pattern;
	pattern = ( pattern << 32 ) | pattern;
	//
	temp u8 mutable_ref bytes_8 = to( u8 mutable_ref, bytes_ref );
	//
	while( size >= 64 )
	{
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		val_of( bytes_8++ ) = pattern;
		size -= 64;
	}
	//
	while( size >= 8 )
	{
		val_of( bytes_8++ ) = pattern;
		size -= 8;
	}
	//
	bytes_ref = to( byte mutable_ref, bytes_8 );
	while( size-- ) { val_of( bytes_ref++ ) = val; }
	//
	out to( byte ref, bytes_ref );
}

embed s4 compare_bytes( mutable byte mutable_ref a_ref,
	mutable byte mutable_ref b_ref, mutable u8 size )
{
	temp const u8 mutable_ref a_8 = to( u8 mutable_ref, a_ref );
	temp const u8 mutable_ref b_8 = to( u8 mutable_ref, b_ref );
	//
	while( size >= 64 )
	{
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		size -= 64;
	}
	//
	while( size >= 8 )
	{
		skip_if( val_of( a_8 ) isnt val_of( b_8 ) );
		a_8++, b_8++;
		size -= 8;
	}
	//
	a_ref = to( byte mutable_ref, a_8 );
	b_ref = to( byte mutable_ref, b_8 );
	while( size-- )
	{
		if( val_of( a_ref ) != val_of( b_ref ) )
		{
			out to( s4, val_of( a_ref ) ) - to( s4, val_of( b_ref ) );
		}
		a_ref++, b_ref++;
	}
	//
	out 0;
}

embed byte ref clear_bytes( mutable byte mutable_ref bytes, mutable u8 size )
{
	out fill_bytes( bytes, 0, size );
}

embed u8 measure_bytes( in byte ref bytes )
{
	if_null( bytes ) { out 0; }
	temp const byte mutable_ref p = bytes;
	while( val_of( p ) ) { p++; }
	out to( u8, p - bytes );
}

#define BYTES_ADD_BUFFER( BYTES ) \
	byte buffer[BYTES]; \
	temp byte mutable_ref p = buffer + BYTES

#define BYTES_ADD_NEG \
	if( val < 0 ) \
	{ \
		val_of( val_of( out_p ) ) = '-'; \
		++val_of( out_p ); \
	}

#define BYTES_ADD_U( VAL ) \
	temp u1 size = 0; \
	do { \
		temp s1 digit = VAL % 10; \
		if( digit < 0 ) digit = -digit; \
		val_of( --p ) = digit + '0'; \
		VAL /= 10, ++size; \
	} while( VAL isnt 0 ); \
	copy_bytes( p, size, val_of( out_p ) ); \
	val_of( out_p ) += size

#define BYTES_ADD_S \
	BYTES_ADD_NEG \
	BYTES_ADD_U

fn bytes_add_byte( mutable byte mutable_ref ref out_p, mutable byte val )
{
	val_of( val_of( out_p ) ) = val;
	val_of( out_p )++;
}

fn bytes_add_u1( mutable byte mutable_ref ref out_p, mutable u1 val )
{
	BYTES_ADD_BUFFER( 3 );
	BYTES_ADD_U( val );
}

fn bytes_add_s1( mutable byte mutable_ref ref out_p, mutable s1 val )
{
	BYTES_ADD_BUFFER( 4 );
	BYTES_ADD_S( val );
}

fn bytes_add_u2( mutable byte mutable_ref ref out_p, mutable u2 val )
{
	BYTES_ADD_BUFFER( 5 );
	BYTES_ADD_U( val );
}

fn bytes_add_s2( mutable byte mutable_ref ref out_p, mutable s2 val )
{
	BYTES_ADD_BUFFER( 6 );
	BYTES_ADD_S( val );
}

fn bytes_add_u4( mutable byte mutable_ref ref out_p, mutable u4 val )
{
	BYTES_ADD_BUFFER( 10 );
	BYTES_ADD_U( val );
}

fn bytes_add_s4( mutable byte mutable_ref ref out_p, mutable s4 val )
{
	BYTES_ADD_BUFFER( 11 );
	BYTES_ADD_S( val );
}

fn bytes_add_u8( mutable byte mutable_ref ref out_p, mutable u8 val )
{
	BYTES_ADD_BUFFER( 20 );
	BYTES_ADD_U( val );
}

fn bytes_add_s8( mutable byte mutable_ref ref out_p, mutable s8 val )
{
	BYTES_ADD_BUFFER( 21 );
	BYTES_ADD_S( val );
}

#define BYTES_ADD_F( N ) \
	temp s##N int_part = to( s##N, val ); \
	temp f##N frac_part = val - int_part; \
	BYTES_ADD_S( int_part ); \
	val_of( val_of( out_p ) ) = '.'; \
	++val_of( out_p ); \
	if( frac_part < 0 ) frac_part = -frac_part; \
	iter( i, N ) \
	{ \
		frac_part *= 10; \
		temp const u1 digit = u1( frac_part ); \
		val_of( val_of( out_p ) ) = '0' + digit; \
		++val_of( out_p ); \
		frac_part -= digit; \
	}

fn bytes_add_f4( mutable byte mutable_ref ref out_p, mutable f4 val )
{
	BYTES_ADD_BUFFER( 32 );
	BYTES_ADD_F( 4 );
}

fn bytes_add_f8( mutable byte mutable_ref ref out_p, mutable f8 val )
{
	BYTES_ADD_BUFFER( 64 );
	BYTES_ADD_F( 8 );
}

#define BYTES_ADD_OCTAL( VAL ) \
	temp u1 size = 0; \
	do { \
		val_of( --p ) = ( VAL & 7 ) + '0'; \
		VAL >>= 3; \
		++size; \
	} while( VAL > 0 ); \
	copy_bytes( p, size, val_of( out_p ) ); \
	val_of( out_p ) += size

fn bytes_add_octal_u1( mutable byte mutable_ref ref out_p, mutable u1 val )
{
	BYTES_ADD_BUFFER( 3 ); // max octal for u1 is 377
	BYTES_ADD_OCTAL( val );
}

fn bytes_add_octal_u2( mutable byte mutable_ref ref out_p, mutable u2 val )
{
	BYTES_ADD_BUFFER( 6 ); // max octal for u2 is 177777
	BYTES_ADD_OCTAL( val );
}

fn bytes_add_octal_u4( mutable byte mutable_ref ref out_p, mutable u4 val )
{
	BYTES_ADD_BUFFER( 11 ); // max octal for u4 is 37777777777
	BYTES_ADD_OCTAL( val );
}

fn bytes_add_octal_u8( mutable byte mutable_ref ref out_p, mutable u8 val )
{
	BYTES_ADD_BUFFER( 22 ); // max octal for u8 is 1777777777777777777777
	BYTES_ADD_OCTAL( val );
}

#define BYTES_ADD_HEX( VAL ) \
	temp u1 size = 0; \
	do { \
		val_of( --p ) = "0123456789ABCDEF"[VAL & 0xF]; \
		VAL >>= 4; \
		++size; \
	} while( VAL > 0 ); \
	copy_bytes( p, size, val_of( out_p ) ); \
	val_of( out_p ) += size

fn bytes_add_hex_u1( mutable byte mutable_ref ref out_p, mutable u1 val )
{
	BYTES_ADD_BUFFER( 2 );
	BYTES_ADD_HEX( val );
}

fn bytes_add_hex_u2( mutable byte mutable_ref ref out_p, mutable u2 val )
{
	BYTES_ADD_BUFFER( 4 );
	BYTES_ADD_HEX( val );
}

fn bytes_add_hex_u4( mutable byte mutable_ref ref out_p, mutable u4 val )
{
	BYTES_ADD_BUFFER( 8 );
	BYTES_ADD_HEX( val );
}

fn bytes_add_hex_u8( mutable byte mutable_ref ref out_p, mutable u8 val )
{
	BYTES_ADD_BUFFER( 16 );
	BYTES_ADD_HEX( val );
}

// // // // // // //

fn format_bytes(
	in byte ref in_bytes, byte mutable_ref out_bytes, mutable va_list _args )
{
	if_null( in_bytes ) out;
	//
	byte mutable_ref out_p = out_bytes;
	temp const byte mutable_ref current = in_bytes - 1;
	//
	loop
	{
		++current;
		leave_if( val_of( current ) is '\0' );
		if( val_of( current ) isnt '|' )
		{
			val_of( out_p ) = val_of( current );
			++out_p;
			skip;
		}
		else
		{
			++current;
			select( val_of( current ) )
			{
				with( 'b' )
				{
					bytes_add_byte( ref_of( out_p ), args_next( byte ) );
					skip;
				}
				//
				with( 'u' )
				{
					++current;
					select( val_of( current ) )
					{
						with( '1' )
						{
							bytes_add_u1( ref_of( out_p ), args_next( u1 ) );
							skip;
						}
						with( '2' )
						{
							bytes_add_u2( ref_of( out_p ), args_next( u2 ) );
							skip;
						}
						with( '4' )
						{
							bytes_add_u4( ref_of( out_p ), args_next( u4 ) );
							skip;
						}
						with( '8' )
						{
							bytes_add_u8( ref_of( out_p ), args_next( u8 ) );
							skip;
						}
						otherwise skip;
					}
				}
				//
				with( 's' )
				{
					++current;
					select( val_of( current ) )
					{
						with( '1' )
						{
							bytes_add_s1( ref_of( out_p ), args_next( s1 ) );
							skip;
						}
						with( '2' )
						{
							bytes_add_s2( ref_of( out_p ), args_next( s2 ) );
							skip;
						}
						with( '4' )
						{
							bytes_add_s4( ref_of( out_p ), args_next( s4 ) );
							skip;
						}
						with( '8' )
						{
							bytes_add_s8( ref_of( out_p ), args_next( s8 ) );
							skip;
						}
						otherwise skip;
					}
				}
				//
				with( 'f' )
				{
					++current;
					select( val_of( current ) )
					{
						with( '4' )
						{
							temp f4 f = args_next( f4 );
							bytes_add_f4( ref_of( out_p ), f );
							skip;
						}
						with( '8' )
						{
							bytes_add_f8( ref_of( out_p ), args_next( f8 ) );
							skip;
						}
						otherwise skip;
					}
				}
				//
				with( 't' )
				{
					byte ref s = to( byte ref, args_next( ref ) );
					temp const u4 size = measure_bytes( s );
					copy_bytes( s, size, out_p );
					out_p += size;
					skip;
				}
				//
				with( 'h' )
				{
					++current;
					select( val_of( current ) )
					{
						with( '1' )
						{
							bytes_add_hex_u1( ref_of( out_p ), args_next( u1 ) );
							skip;
						}
						with( '2' )
						{
							bytes_add_hex_u2( ref_of( out_p ), args_next( u2 ) );
							skip;
						}
						with( '4' )
						{
							bytes_add_hex_u4( ref_of( out_p ), args_next( u4 ) );
							skip;
						}
						with( '8' )
						{
							bytes_add_hex_u8( ref_of( out_p ), args_next( u8 ) );
							skip;
						}
						otherwise skip;
					}
				}
				//
				with( 'o' )
				{
					++current;
					select( val_of( current ) )
					{
						with( '1' )
						{
							bytes_add_octal_u1( ref_of( out_p ), args_next( u1 ) );
							skip;
						}
						with( '2' )
						{
							bytes_add_octal_u2( ref_of( out_p ), args_next( u2 ) );
							skip;
						}
						with( '4' )
						{
							bytes_add_octal_u4( ref_of( out_p ), args_next( u4 ) );
							skip;
						}
						with( '8' )
						{
							bytes_add_octal_u8( ref_of( out_p ), args_next( u8 ) );
							skip;
						}
						otherwise skip;
					}
				}
			}
		}
	}
	//
	val_of( out_p ) = '\0';
}

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///

extern anon mutable_ref malloc( u8 size );
extern anon mutable_ref calloc( u8 nmemb, u8 size );
extern anon mutable_ref realloc( anon mutable_ref ptr, u8 size );
extern anon free( anon mutable_ref ptr );

embed byte ref _new_ref( u4 typesize, byte ref default_val, u8 size )
{
	temp byte ref r = to( byte ref, calloc( size, typesize ) );
	if( default_val ) { copy_bytes( default_val, size * typesize, r ); }
	out r;
}

#define new_ref( TYPE, ... ) \
	to( TYPE ref, \
		_new_ref( size_of( TYPE ), DEFAULTS( ( null, 1 ), __VA_ARGS__ ) ) )

embed byte ref _resize_ref( byte ref r, in u8 old_size, in u8 new_size )
{
	temp byte ref resized = to( byte ref, realloc( r, new_size ) );
	if( new_size > old_size )
	{
		clear_bytes( resized + old_size, ( new_size - old_size ) );
	}
	out resized;
}

#define resize_ref( REF, REF_TYPE, OLD_BYTES_SIZE, NEW_BYTES_SIZE ) \
	to( REF_TYPE ref, \
		_resize_ref( to( byte ref, REF ), OLD_BYTES_SIZE, NEW_BYTES_SIZE ) )
#define delete_ref( REF_TYPE ) free( REF_TYPE )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// object (struct reference)

#define object( ... ) \
	_struct_make( __VA_ARGS__, mutable_ref, \
		fn_type( anon, __VA_ARGS__ ) JOIN( __VA_ARGS__, _fn ); )
#define object_fn( OBJ, FN ) \
	fn _##FN( in OBJ this ); \
	OBJ##_fn FN = to( OBJ##_fn, _##FN ); \
	fn _##FN( in OBJ this )
#define call( OBJ, FN ) \
	if( OBJ->FN isnt null ) OBJ->FN( OBJ )

#define new_object( TYPE, ... ) \
	new_ref( struct TYPE, \
		PASTE_IF_ARGS( to( byte ref, ref_of( make( struct TYPE, __VA_ARGS__ ) ) ), \
			__VA_ARGS__ ), \
		size_of( struct TYPE ) )

#define delete_object( OBJ ) delete_ref( OBJ )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// dynamic array structure

object( list )
{
	byte mutable_ref data;
	u4 size;
	u4 capacity;
	u1 type_size;
};

#define new_list_size( TYPE, BYTES, SIZE ) \
	new_object( list, \
		.data = new_ref( byte, BYTES, ( SIZE + 1 ) * size_of( TYPE ) ), \
		.size = SIZE, .capacity = ( SIZE + 1 ), .type_size = size_of( TYPE ) )

#define new_list( TYPE ) new_list_size( TYPE, null, 0 )

#define list_duplicate( LIST, TYPE ) \
	new_list_size( TYPE, LIST->data, LIST->size )

#define _grow_list( LIST, MIN_SIZE ) \
	if( LIST->size >= LIST->capacity ) \
	{ \
		temp u4 doubled = LIST->capacity << 1; \
		temp u4 needed = ( MIN_SIZE ); \
		temp u4 new_cap = pick( needed > doubled, needed, doubled ); \
		LIST->data = \
			resize_ref( LIST->data, byte, ( LIST->capacity + 1 ) * LIST->type_size, \
				( new_cap + 1 ) * LIST->type_size ); \
		LIST->capacity = new_cap; \
	}

#define list_get( LIST, TYPE, POS ) ( to( TYPE ref, LIST->data ) )[POS]
#define list_set( LIST, TYPE, POS, VAL ) \
	( to( TYPE ref, LIST->data ) )[POS] = ( VAL )
#define list_front( LIST, TYPE ) list_get( LIST, TYPE, 0 )
#define list_back( LIST, TYPE ) list_get( LIST, TYPE, LIST->size - 1 )

#define list_add( LIST, TYPE, VAL ) \
	DEF_START \
	_grow_list( LIST, LIST->size ); \
	( to( TYPE ref, LIST->data ) )[LIST->size++] = ( VAL ); \
	DEF_END

#define list_insert( LIST, TYPE, POS, VAL ) \
	DEF_START \
	_grow_list( LIST, LIST->size ); \
	temp byte mutable_ref src = LIST->data + ( POS * LIST->type_size ); \
	temp u4 move_size = ( LIST->size - POS ) * LIST->type_size; \
	move_bytes( src, move_size, src + LIST->type_size ); \
	( to( TYPE ref, LIST->data ) )[POS] = ( VAL ); \
	++LIST->size; \
	DEF_END

#define list_add_bytes( LIST, BYTES, SIZE ) \
	DEF_START \
	temp u4 _SIZE = SIZE; \
	_grow_list( LIST, LIST->size + _SIZE ); \
	temp byte mutable_ref dest = LIST->data + ( LIST->size * LIST->type_size ); \
	copy_bytes( BYTES, _SIZE, dest ); \
	LIST->size += _SIZE; \
	DEF_END

#define list_insert_bytes( LIST, POS, BYTES, SIZE ) \
	DEF_START \
	temp u4 _SIZE = SIZE; \
	_grow_list( LIST, LIST->size + _SIZE ); \
	temp byte mutable_ref src = LIST->data + ( POS * LIST->type_size ); \
	temp u4 move_size = ( LIST->size - POS ) * LIST->type_size; \
	move_bytes( src, move_size, src + ( _SIZE * LIST->type_size ) ); \
	copy_bytes( BYTES, _SIZE, src ); \
	LIST->size += _SIZE; \
	DEF_END

#define list_delete( LIST, POS ) \
	DEF_START \
	move_bytes( LIST->data + ( ( POS + 1 ) * LIST->type_size ), \
		( LIST->size - POS - 1 ) * LIST->type_size, \
		LIST->data + ( POS * LIST->type_size ) ); \
	--LIST->size; \
	DEF_END

#define list_remove_first( LIST, TYPE ) \
	list_get( LIST, TYPE, 0 ); \
	list_delete( LIST, 0 )
#define list_remove_last( LIST, TYPE ) list_get( LIST, TYPE, --LIST->size )
#define clear_listLIST LIST->size = 0
#define delete_listLIST \
	DEF_START \
	delete_ref( LIST->data ); \
	delete_refLIST; \
	DEF_END

// // // // // // //
// text

type_from( list ) text;

#define new_text() new_list( byte )
#define new_text_size( BYTES, SIZE ) new_list_size( byte, BYTES, SIZE )
#define text_duplicate( TEXT ) list_duplicate( TEXT, byte )

#define text_terminate( TEXT ) ( ( TEXT )->data[( TEXT )->size] = 0 )

#define text_get( TEXT, POS ) list_get( TEXT, byte, POS )
#define text_set( TEXT, POS, CHAR ) list_set( TEXT, byte, POS, CHAR )
#define text_front( TEXT ) list_front( TEXT, byte )
#define text_back( TEXT ) list_back( TEXT, byte )

#define text_add( TEXT, CHAR ) list_add( TEXT, byte, CHAR );
#define text_insert( TEXT, POS, CHAR ) list_insert( TEXT, byte, POS, CHAR );
#define text_add_bytes( TEXT, BYTES, ... ) \
	list_add_bytes( \
		TEXT, BYTES, DEFAULTS( ( measure_bytes( BYTES ) ), __VA_ARGS__ ) );
#define text_insert_bytes( TEXT, POS, BYTES, SIZE ) \
	list_insert_bytes( TEXT, POS, BYTES, SIZE );

#define text_delete( TEXT, POS ) list_delete( TEXT, POS );
#define text_remove_first( TEXT ) list_remove_first( TEXT, byte )
#define text_remove_last( TEXT ) list_remove_last( TEXT, byte )

#define text_newline( TEXT ) text_add( TEXT, '\n' )
#define text_clear( TEXT ) \
	DEF_START \
	clear_list( TEXT ); \
	text_terminate( TEXT ); \
	DEF_END
#define delete_text( TEXT ) delete_list( TEXT )

#define text_resize( TEXT, SIZE ) \
	DEF_START \
	if( ( TEXT )->capacity <= SIZE + 1 ) _grow_list( TEXT, SIZE ); \
	( TEXT )->size = SIZE; \
	text_terminate( TEXT ); \
	DEF_END

#define text_add_text( TEXT, OTHER ) \
	DEF_START \
	list_add_bytes( TEXT, ( OTHER )->data, ( OTHER )->size ); \
	text_terminate( TEXT ); \
	DEF_END

#define text_insert_text( TEXT, POS, OTHER ) \
	DEF_START \
	list_insert_bytes( TEXT, POS, ( OTHER )->data, ( OTHER )->size ); \
	text_terminate( TEXT ); \
	DEF_END

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// pile

object( pile )
{
	u4 count;
	u4 add_pos;
	list data;
	list deleted;
};

#define new_pile( TYPE ) \
	new_object( pile, .count = 0, .add_pos = 0, .data = new_list( TYPE ), \
		.deleted = new_list( u4 ) )

#define pile_add( PILE, TYPE, VAL ) \
	DEF_START \
	++( PILE )->count; \
	if( ( PILE )->deleted->size ) \
	{ \
		( PILE )->add_pos = list_remove_last( ( PILE )->deleted, u4 ); \
		list_set( ( PILE )->data, TYPE, ( PILE )->add_pos, VAL ); \
	} \
	else \
	{ \
		( PILE )->add_pos = ( PILE )->data->size; \
		list_add( ( PILE )->data, TYPE, VAL ); \
	} \
	DEF_END

#define pile_get( PILE, TYPE, POS ) list_get( ( PILE )->data, TYPE, POS )

#define pile_delete( PILE, TYPE, POS ) \
	DEF_START \
	--( PILE )->count; \
	list_add( ( PILE )->deleted, u4, POS ); \
	list_set( ( PILE )->data, TYPE, POS, make( TYPE, 0 ) ); \
	DEF_END

#define delete_pile( PILE ) \
	DEF_START \
	delete_list( ( PILE )->data ); \
	delete_list( ( PILE )->deleted ); \
	free( PILE ); \
	DEF_END

#define iter_pile( PILE, POS ) \
	temp const pile PILE_##POS = ( PILE ); \
	temp u4 POS##_data_pos = 0; \
	temp u4 POS##_count = ( PILE )->count; \
	iter( POS, POS##_count )

#define get_pile_iter( TYPE, VAL, POS, SKIP ) \
	temp const TYPE VAL = pile_get( PILE_##POS, TYPE, POS##_data_pos++ ); \
	if( SKIP ) \
	{ \
		--POS; \
		skip; \
	}

#define get_pile_iter_ref( POS, TYPE, SKIP ) \
	temp TYPE ref POS##_##TYPE##_ref = \
		ref_of( pile_get( PILE_##POS, TYPE, POS##_data_pos++ ) ); \
	if( SKIP ) \
	{ \
		--POS; \
		skip; \
	}

#define pile_iter_delete( TYPE, POS ) \
	pile_delete( PILE_##POS, TYPE, POS##_data_pos - 1 )

#define pile_is_valid( PILE, POS ) \
	( POS < ( PILE )->data->size && !list_contains( ( PILE )->deleted, u4, POS ) )

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
/// main executable scope

#define main_scope int main()

/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
#endif // H_LANG
/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///
