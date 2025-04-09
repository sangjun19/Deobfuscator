	.file	"fboxf_C_A_Modern_Approach_Programming_Projects_11_flatten.c"
	.text
	.globl	_TIG_IZ_M5BK_argv
	.bss
	.align 8
	.type	_TIG_IZ_M5BK_argv, @object
	.size	_TIG_IZ_M5BK_argv, 8
_TIG_IZ_M5BK_argv:
	.zero	8
	.globl	_TIG_IZ_M5BK_envp
	.align 8
	.type	_TIG_IZ_M5BK_envp, @object
	.size	_TIG_IZ_M5BK_envp, 8
_TIG_IZ_M5BK_envp:
	.zero	8
	.globl	_TIG_IZ_M5BK_argc
	.align 4
	.type	_TIG_IZ_M5BK_argc, @object
	.size	_TIG_IZ_M5BK_argc, 4
_TIG_IZ_M5BK_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter phone number: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_M5BK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_M5BK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_M5BK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 153 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-M5BK--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_M5BK_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_M5BK_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_M5BK_envp(%rip)
	nop
	movq	$6, -40(%rbp)
.L58:
	cmpq	$44, -40(%rbp)
	ja	.L61
	movq	-40(%rbp), %rax
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
	.long	.L34-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L61-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L61-.L8
	.long	.L26-.L8
	.long	.L61-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L61-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L61-.L8
	.long	.L12-.L8
	.long	.L61-.L8
	.long	.L11-.L8
	.long	.L61-.L8
	.long	.L10-.L8
	.long	.L61-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L23:
	call	getchar@PLT
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movb	%al, -69(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L35
.L32:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -64(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L35
.L26:
	movl	-64(%rbp), %eax
	cltq
	movb	$54, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L27:
	movq	$8, -40(%rbp)
	jmp	.L35
.L29:
	addl	$1, -64(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L35
.L19:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-69(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L36
	movq	$21, -40(%rbp)
	jmp	.L35
.L36:
	movq	$8, -40(%rbp)
	jmp	.L35
.L33:
	cmpb	$10, -69(%rbp)
	je	.L38
	movq	$33, -40(%rbp)
	jmp	.L35
.L38:
	movq	$17, -40(%rbp)
	jmp	.L35
.L25:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L35
.L18:
	movb	$0, -32(%rbp)
	movl	$1, -68(%rbp)
	movq	$28, -40(%rbp)
	jmp	.L35
.L21:
	movsbl	-69(%rbp), %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -60(%rbp)
	movq	$22, -40(%rbp)
	jmp	.L35
.L28:
	movl	-64(%rbp), %eax
	cltq
	movb	$51, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L14:
	movl	-68(%rbp), %eax
	movb	$0, -32(%rbp,%rax)
	addl	$1, -68(%rbp)
	movq	$28, -40(%rbp)
	jmp	.L35
.L24:
	movl	$0, -56(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L35
.L31:
	movq	$24, -40(%rbp)
	jmp	.L35
.L17:
	movl	-64(%rbp), %eax
	cltq
	movb	$57, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L20:
	movl	-60(%rbp), %eax
	subl	$65, %eax
	cmpl	$24, %eax
	ja	.L40
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L42(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L42(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L42:
	.long	.L49-.L42
	.long	.L49-.L42
	.long	.L49-.L42
	.long	.L48-.L42
	.long	.L48-.L42
	.long	.L48-.L42
	.long	.L47-.L42
	.long	.L47-.L42
	.long	.L47-.L42
	.long	.L46-.L42
	.long	.L46-.L42
	.long	.L46-.L42
	.long	.L45-.L42
	.long	.L45-.L42
	.long	.L45-.L42
	.long	.L44-.L42
	.long	.L40-.L42
	.long	.L44-.L42
	.long	.L44-.L42
	.long	.L43-.L42
	.long	.L43-.L42
	.long	.L43-.L42
	.long	.L41-.L42
	.long	.L41-.L42
	.long	.L41-.L42
	.text
.L41:
	movq	$27, -40(%rbp)
	jmp	.L50
.L43:
	movq	$29, -40(%rbp)
	jmp	.L50
.L44:
	movq	$0, -40(%rbp)
	jmp	.L50
.L45:
	movq	$14, -40(%rbp)
	jmp	.L50
.L46:
	movq	$7, -40(%rbp)
	jmp	.L50
.L47:
	movq	$39, -40(%rbp)
	jmp	.L50
.L48:
	movq	$11, -40(%rbp)
	jmp	.L50
.L49:
	movq	$43, -40(%rbp)
	jmp	.L50
.L40:
	movq	$12, -40(%rbp)
	nop
.L50:
	jmp	.L35
.L16:
	cmpl	$19, -68(%rbp)
	jbe	.L51
	movq	$4, -40(%rbp)
	jmp	.L35
.L51:
	movq	$32, -40(%rbp)
	jmp	.L35
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L59
	jmp	.L60
.L13:
	cmpl	$19, -64(%rbp)
	jg	.L54
	movq	$16, -40(%rbp)
	jmp	.L35
.L54:
	movq	$17, -40(%rbp)
	jmp	.L35
.L12:
	movl	-56(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -56(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L35
.L10:
	movl	$10, %edi
	call	putchar@PLT
	movq	$44, -40(%rbp)
	jmp	.L35
.L34:
	movl	-64(%rbp), %eax
	cltq
	movb	$55, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L11:
	movl	-64(%rbp), %eax
	cltq
	movb	$52, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L30:
	movl	-64(%rbp), %eax
	cltq
	movb	$53, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L15:
	movl	-64(%rbp), %eax
	cltq
	movb	$56, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L9:
	movl	-64(%rbp), %eax
	cltq
	movb	$50, -32(%rbp,%rax)
	movq	$8, -40(%rbp)
	jmp	.L35
.L22:
	cmpl	$19, -56(%rbp)
	jg	.L56
	movq	$37, -40(%rbp)
	jmp	.L35
.L56:
	movq	$41, -40(%rbp)
	jmp	.L35
.L61:
	nop
.L35:
	jmp	.L58
.L60:
	call	__stack_chk_fail@PLT
.L59:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
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
