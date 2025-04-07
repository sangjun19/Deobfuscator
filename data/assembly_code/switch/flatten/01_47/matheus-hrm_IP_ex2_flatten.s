	.file	"matheus-hrm_IP_ex2_flatten.c"
	.text
	.globl	_TIG_IZ_xIlc_argc
	.bss
	.align 4
	.type	_TIG_IZ_xIlc_argc, @object
	.size	_TIG_IZ_xIlc_argc, 4
_TIG_IZ_xIlc_argc:
	.zero	4
	.globl	_TIG_IZ_xIlc_argv
	.align 8
	.type	_TIG_IZ_xIlc_argv, @object
	.size	_TIG_IZ_xIlc_argv, 8
_TIG_IZ_xIlc_argv:
	.zero	8
	.globl	_TIG_IZ_xIlc_envp
	.align 8
	.type	_TIG_IZ_xIlc_envp, @object
	.size	_TIG_IZ_xIlc_envp, 8
_TIG_IZ_xIlc_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-98304(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$1760, %rsp
	movl	%edi, -100036(%rbp)
	movq	%rsi, -100048(%rbp)
	movq	%rdx, -100056(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_xIlc_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_xIlc_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_xIlc_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-xIlc--0
# 0 "" 2
#NO_APP
	movl	-100036(%rbp), %eax
	movl	%eax, _TIG_IZ_xIlc_argc(%rip)
	movq	-100048(%rbp), %rax
	movq	%rax, _TIG_IZ_xIlc_argv(%rip)
	movq	-100056(%rbp), %rax
	movq	%rax, _TIG_IZ_xIlc_envp(%rip)
	nop
	movq	$8, -100024(%rbp)
.L17:
	cmpq	$8, -100024(%rbp)
	ja	.L20
	movq	-100024(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L20-.L8
	.long	.L10-.L8
	.long	.L20-.L8
	.long	.L20-.L8
	.long	.L20-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	-100032(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -100024(%rbp)
	jmp	.L13
.L11:
	cmpl	$0, -100028(%rbp)
	je	.L14
	movq	$0, -100024(%rbp)
	jmp	.L13
.L14:
	movq	$3, -100024(%rbp)
	jmp	.L13
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L12:
	leaq	-100016(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-100016(%rbp), %rax
	movq	%rax, %rdi
	call	verifica
	movq	$7, -100024(%rbp)
	jmp	.L13
.L9:
	movl	-100032(%rbp), %eax
	movl	%eax, -100028(%rbp)
	movl	-100032(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -100032(%rbp)
	movq	$1, -100024(%rbp)
	jmp	.L13
.L20:
	nop
.L13:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	led
	.type	led, @function
led:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$15, -8(%rbp)
.L52:
	cmpq	$21, -8(%rbp)
	ja	.L54
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L36-.L24
	.long	.L35-.L24
	.long	.L54-.L24
	.long	.L54-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L54-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L54-.L24
	.long	.L29-.L24
	.long	.L54-.L24
	.long	.L54-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L54-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L54-.L24
	.long	.L54-.L24
	.long	.L23-.L24
	.text
.L25:
	movl	$6, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L34:
	movl	$6, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L28:
	movl	$4, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L27:
	movsbl	-20(%rbp), %eax
	subl	$48, %eax
	cmpl	$9, %eax
	ja	.L38
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L49-.L40
	.long	.L48-.L40
	.long	.L47-.L40
	.long	.L46-.L40
	.long	.L45-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L41-.L40
	.long	.L39-.L40
	.text
.L41:
	movq	$1, -8(%rbp)
	jmp	.L50
.L42:
	movq	$8, -8(%rbp)
	jmp	.L50
.L49:
	movq	$18, -8(%rbp)
	jmp	.L50
.L39:
	movq	$0, -8(%rbp)
	jmp	.L50
.L43:
	movq	$4, -8(%rbp)
	jmp	.L50
.L45:
	movq	$14, -8(%rbp)
	jmp	.L50
.L44:
	movq	$7, -8(%rbp)
	jmp	.L50
.L46:
	movq	$21, -8(%rbp)
	jmp	.L50
.L47:
	movq	$9, -8(%rbp)
	jmp	.L50
.L48:
	movq	$11, -8(%rbp)
	jmp	.L50
.L38:
	movq	$5, -8(%rbp)
	nop
.L50:
	jmp	.L37
.L31:
	movl	$3, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L35:
	movl	$7, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L23:
	movl	$5, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L29:
	movl	$2, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L30:
	movl	$5, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L26:
	movl	-12(%rbp), %eax
	jmp	.L53
.L33:
	movq	$17, -8(%rbp)
	jmp	.L37
.L36:
	movl	$6, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L32:
	movl	$5, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L37
.L54:
	nop
.L37:
	jmp	.L52
.L53:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	led, .-led
	.section	.rodata
.LC2:
	.string	"%d leds\n"
	.text
	.globl	verifica
	.type	verifica, @function
verifica:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$6, -16(%rbp)
.L68:
	cmpq	$8, -16(%rbp)
	ja	.L69
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L58(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L58(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L58:
	.long	.L63-.L58
	.long	.L69-.L58
	.long	.L69-.L58
	.long	.L62-.L58
	.long	.L61-.L58
	.long	.L69-.L58
	.long	.L60-.L58
	.long	.L70-.L58
	.long	.L57-.L58
	.text
.L61:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, -32(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -28(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L64
.L57:
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	led
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	addl	%eax, -24(%rbp)
	addl	$1, -28(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L64
.L62:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L64
.L60:
	movq	$4, -16(%rbp)
	jmp	.L64
.L63:
	movl	-28(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L65
	movq	$8, -16(%rbp)
	jmp	.L64
.L65:
	movq	$3, -16(%rbp)
	jmp	.L64
.L69:
	nop
.L64:
	jmp	.L68
.L70:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	verifica, .-verifica
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
