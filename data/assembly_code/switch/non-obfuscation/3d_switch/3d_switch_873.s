	.file	"3d_switch_873.c"
	.def	___main;	.scl	2;	.type	32;	.endef
	.section .rdata,"dr"
LC0:
	.ascii "1-1-1\0"
LC1:
	.ascii "1-1-2\0"
LC2:
	.ascii "1-1-3\0"
LC3:
	.ascii "1-2-1\0"
LC4:
	.ascii "1-2-2\0"
LC5:
	.ascii "1-2-3\0"
LC6:
	.ascii "1-3-1\0"
LC7:
	.ascii "1-3-2\0"
LC8:
	.ascii "1-3-3\0"
LC9:
	.ascii "2-1-1\0"
LC10:
	.ascii "2-1-2\0"
LC11:
	.ascii "2-1-3\0"
LC12:
	.ascii "2-2-1\0"
LC13:
	.ascii "2-2-2\0"
LC14:
	.ascii "2-2-3\0"
LC15:
	.ascii "2-3-1\0"
LC16:
	.ascii "2-3-2\0"
LC17:
	.ascii "2-3-3\0"
LC18:
	.ascii "3-1-1\0"
LC19:
	.ascii "3-1-2\0"
LC20:
	.ascii "3-1-3\0"
LC21:
	.ascii "3-2-1\0"
LC22:
	.ascii "3-2-2\0"
LC23:
	.ascii "3-2-3\0"
LC24:
	.ascii "3-3-1\0"
LC25:
	.ascii "3-3-2\0"
LC26:
	.ascii "3-3-3\0"
	.text
	.globl	_main
	.def	_main;	.scl	2;	.type	32;	.endef
_main:
LFB10:
	.cfi_startproc
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register 5
	andl	$-16, %esp
	subl	$32, %esp
	call	___main
	movl	$97, 28(%esp)
	movl	$3, 24(%esp)
	movl	$3, 20(%esp)
	movl	28(%esp), %eax
	cmpl	$2, %eax
	je	L3
	cmpl	$3, %eax
	je	L4
	cmpl	$1, %eax
	jne	L2
	movl	24(%esp), %eax
	cmpl	$2, %eax
	je	L7
	cmpl	$3, %eax
	je	L8
	cmpl	$1, %eax
	je	L9
	jmp	L2
L9:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L11
	cmpl	$3, %eax
	je	L12
	cmpl	$1, %eax
	je	L13
	jmp	L6
L13:
	movl	$LC0, (%esp)
	call	_puts
	jmp	L10
L11:
	movl	$LC1, (%esp)
	call	_puts
	jmp	L10
L12:
	movl	$LC2, (%esp)
	call	_puts
	nop
L10:
	jmp	L6
L7:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L15
	cmpl	$3, %eax
	je	L16
	cmpl	$1, %eax
	je	L17
	jmp	L6
L17:
	movl	$LC3, (%esp)
	call	_puts
	jmp	L14
L15:
	movl	$LC4, (%esp)
	call	_puts
	jmp	L14
L16:
	movl	$LC5, (%esp)
	call	_puts
	nop
L14:
	jmp	L6
L8:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L19
	cmpl	$3, %eax
	je	L20
	cmpl	$1, %eax
	je	L21
	jmp	L55
L21:
	movl	$LC6, (%esp)
	call	_puts
	jmp	L18
L19:
	movl	$LC7, (%esp)
	call	_puts
	jmp	L18
L20:
	movl	$LC8, (%esp)
	call	_puts
	nop
L18:
L55:
	nop
L6:
	jmp	L2
L3:
	movl	24(%esp), %eax
	cmpl	$2, %eax
	je	L23
	cmpl	$3, %eax
	je	L24
	cmpl	$1, %eax
	je	L25
	jmp	L2
L25:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L27
	cmpl	$3, %eax
	je	L28
	cmpl	$1, %eax
	je	L29
	jmp	L22
L29:
	movl	$LC9, (%esp)
	call	_puts
	jmp	L26
L27:
	movl	$LC10, (%esp)
	call	_puts
	jmp	L26
L28:
	movl	$LC11, (%esp)
	call	_puts
	nop
L26:
	jmp	L22
L23:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L31
	cmpl	$3, %eax
	je	L32
	cmpl	$1, %eax
	je	L33
	jmp	L22
L33:
	movl	$LC12, (%esp)
	call	_puts
	jmp	L30
L31:
	movl	$LC13, (%esp)
	call	_puts
	jmp	L30
L32:
	movl	$LC14, (%esp)
	call	_puts
	nop
L30:
	jmp	L22
L24:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L35
	cmpl	$3, %eax
	je	L36
	cmpl	$1, %eax
	je	L37
	jmp	L56
L37:
	movl	$LC15, (%esp)
	call	_puts
	jmp	L34
L35:
	movl	$LC16, (%esp)
	call	_puts
	jmp	L34
L36:
	movl	$LC17, (%esp)
	call	_puts
	nop
L34:
L56:
	nop
L22:
	jmp	L2
L4:
	movl	24(%esp), %eax
	cmpl	$2, %eax
	je	L39
	cmpl	$3, %eax
	je	L40
	cmpl	$1, %eax
	je	L41
	jmp	L58
L41:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L43
	cmpl	$3, %eax
	je	L44
	cmpl	$1, %eax
	je	L45
	jmp	L38
L45:
	movl	$LC18, (%esp)
	call	_puts
	jmp	L42
L43:
	movl	$LC19, (%esp)
	call	_puts
	jmp	L42
L44:
	movl	$LC20, (%esp)
	call	_puts
	nop
L42:
	jmp	L38
L39:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L47
	cmpl	$3, %eax
	je	L48
	cmpl	$1, %eax
	je	L49
	jmp	L38
L49:
	movl	$LC21, (%esp)
	call	_puts
	jmp	L46
L47:
	movl	$LC22, (%esp)
	call	_puts
	jmp	L46
L48:
	movl	$LC23, (%esp)
	call	_puts
	nop
L46:
	jmp	L38
L40:
	movl	20(%esp), %eax
	cmpl	$2, %eax
	je	L51
	cmpl	$3, %eax
	je	L52
	cmpl	$1, %eax
	je	L53
	jmp	L57
L53:
	movl	$LC24, (%esp)
	call	_puts
	jmp	L50
L51:
	movl	$LC25, (%esp)
	call	_puts
	jmp	L50
L52:
	movl	$LC26, (%esp)
	call	_puts
	nop
L50:
L57:
	nop
L38:
L58:
	nop
L2:
	movl	$0, %eax
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
LFE10:
	.ident	"GCC: (MinGW.org GCC-6.3.0-1) 6.3.0"
	.def	_puts;	.scl	2;	.type	32;	.endef
