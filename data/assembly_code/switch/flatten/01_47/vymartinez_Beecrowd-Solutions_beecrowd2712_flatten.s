	.file	"vymartinez_Beecrowd-Solutions_beecrowd2712_flatten.c"
	.text
	.globl	_TIG_IZ_dUrT_argv
	.bss
	.align 8
	.type	_TIG_IZ_dUrT_argv, @object
	.size	_TIG_IZ_dUrT_argv, 8
_TIG_IZ_dUrT_argv:
	.zero	8
	.globl	_TIG_IZ_dUrT_argc
	.align 4
	.type	_TIG_IZ_dUrT_argc, @object
	.size	_TIG_IZ_dUrT_argc, 4
_TIG_IZ_dUrT_argc:
	.zero	4
	.globl	_TIG_IZ_dUrT_envp
	.align 8
	.type	_TIG_IZ_dUrT_envp, @object
	.size	_TIG_IZ_dUrT_envp, 8
_TIG_IZ_dUrT_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%s"
.LC1:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_dUrT_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_dUrT_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_dUrT_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 112 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dUrT--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_dUrT_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_dUrT_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_dUrT_envp(%rip)
	nop
	movq	$26, -88(%rbp)
.L91:
	cmpq	$77, -88(%rbp)
	ja	.L94
	movq	-88(%rbp), %rax
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
	.long	.L94-.L8
	.long	.L52-.L8
	.long	.L94-.L8
	.long	.L51-.L8
	.long	.L50-.L8
	.long	.L49-.L8
	.long	.L48-.L8
	.long	.L47-.L8
	.long	.L46-.L8
	.long	.L45-.L8
	.long	.L94-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L94-.L8
	.long	.L42-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L41-.L8
	.long	.L40-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L39-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L94-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L94-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L25-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L24-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L21-.L8
	.long	.L94-.L8
	.long	.L20-.L8
	.long	.L94-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L94-.L8
	.long	.L13-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L94-.L8
	.long	.L10-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L40:
	movl	$5, -132(%rbp)
	movq	$47, -88(%rbp)
	jmp	.L53
.L23:
	cmpl	$9, -112(%rbp)
	jbe	.L54
	movq	$44, -88(%rbp)
	jmp	.L53
.L54:
	movq	$76, -88(%rbp)
	jmp	.L53
.L38:
	movl	-116(%rbp), %eax
	movb	$0, -40(%rbp,%rax)
	addl	$1, -116(%rbp)
	movq	$39, -88(%rbp)
	jmp	.L53
.L50:
	movl	$0, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L34:
	movl	$5, -132(%rbp)
	movq	$47, -88(%rbp)
	jmp	.L53
.L15:
	leaq	-18(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -96(%rbp)
	movq	$41, -88(%rbp)
	jmp	.L53
.L42:
	movl	$2, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L20:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L92
	jmp	.L93
.L33:
	movb	$70, -30(%rbp)
	movb	$65, -29(%rbp)
	movb	$73, -28(%rbp)
	movb	$76, -27(%rbp)
	movb	$85, -26(%rbp)
	movb	$82, -25(%rbp)
	movb	$69, -24(%rbp)
	movb	$0, -23(%rbp)
	movl	$8, -112(%rbp)
	movq	$50, -88(%rbp)
	jmp	.L53
.L43:
	movb	$87, -60(%rbp)
	movb	$69, -59(%rbp)
	movb	$68, -58(%rbp)
	movb	$78, -57(%rbp)
	movb	$69, -56(%rbp)
	movb	$83, -55(%rbp)
	movb	$68, -54(%rbp)
	movb	$65, -53(%rbp)
	movb	$89, -52(%rbp)
	movb	$0, -51(%rbp)
	movb	$84, -50(%rbp)
	movb	$72, -49(%rbp)
	movb	$85, -48(%rbp)
	movb	$82, -47(%rbp)
	movb	$83, -46(%rbp)
	movb	$68, -45(%rbp)
	movb	$65, -44(%rbp)
	movb	$89, -43(%rbp)
	movb	$0, -42(%rbp)
	movl	$9, -120(%rbp)
	movq	$21, -88(%rbp)
	jmp	.L53
.L11:
	movzbl	-11(%rbp), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	cmpl	$9, %eax
	ja	.L57
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L58-.L59
	.long	.L63-.L59
	.long	.L63-.L59
	.long	.L62-.L59
	.long	.L62-.L59
	.long	.L61-.L59
	.long	.L61-.L59
	.long	.L60-.L59
	.long	.L60-.L59
	.long	.L58-.L59
	.text
.L58:
	movq	$63, -88(%rbp)
	jmp	.L64
.L60:
	movq	$71, -88(%rbp)
	jmp	.L64
.L61:
	movq	$14, -88(%rbp)
	jmp	.L64
.L62:
	movq	$54, -88(%rbp)
	jmp	.L64
.L63:
	movq	$4, -88(%rbp)
	jmp	.L64
.L57:
	movq	$1, -88(%rbp)
	nop
.L64:
	jmp	.L53
.L46:
	movb	$77, -80(%rbp)
	movb	$79, -79(%rbp)
	movb	$78, -78(%rbp)
	movb	$68, -77(%rbp)
	movb	$65, -76(%rbp)
	movb	$89, -75(%rbp)
	movb	$0, -74(%rbp)
	movl	$7, -128(%rbp)
	movq	$40, -88(%rbp)
	jmp	.L53
.L21:
	movl	$1, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L52:
	movl	$5, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L7:
	movl	$5, -132(%rbp)
	movq	$6, -88(%rbp)
	jmp	.L53
.L51:
	movl	-136(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jge	.L65
	movq	$62, -88(%rbp)
	jmp	.L53
.L65:
	movq	$56, -88(%rbp)
	jmp	.L53
.L39:
	cmpl	$9, -120(%rbp)
	jbe	.L67
	movq	$17, -88(%rbp)
	jmp	.L53
.L67:
	movq	$58, -88(%rbp)
	jmp	.L53
.L29:
	movzbl	-15(%rbp), %eax
	cmpb	$45, %al
	je	.L69
	movq	$65, -88(%rbp)
	jmp	.L53
.L69:
	movq	$6, -88(%rbp)
	jmp	.L53
.L9:
	movl	-112(%rbp), %eax
	movb	$0, -30(%rbp,%rax)
	addl	$1, -112(%rbp)
	movq	$50, -88(%rbp)
	jmp	.L53
.L12:
	movl	-124(%rbp), %eax
	movb	$0, -70(%rbp,%rax)
	addl	$1, -124(%rbp)
	movq	$61, -88(%rbp)
	jmp	.L53
.L37:
	movq	$8, -88(%rbp)
	jmp	.L53
.L44:
	movl	-104(%rbp), %eax
	cltq
	movzbl	-18(%rbp,%rax), %eax
	cmpb	$64, %al
	jg	.L71
	movq	$30, -88(%rbp)
	jmp	.L53
.L71:
	movq	$5, -88(%rbp)
	jmp	.L53
.L45:
	movl	$4, -100(%rbp)
	movq	$7, -88(%rbp)
	jmp	.L53
.L14:
	movl	$4, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L22:
	addl	$1, -100(%rbp)
	movq	$7, -88(%rbp)
	jmp	.L53
.L41:
	movb	$70, -40(%rbp)
	movb	$82, -39(%rbp)
	movb	$73, -38(%rbp)
	movb	$68, -37(%rbp)
	movb	$65, -36(%rbp)
	movb	$89, -35(%rbp)
	movb	$0, -34(%rbp)
	movl	$7, -116(%rbp)
	movq	$39, -88(%rbp)
	jmp	.L53
.L27:
	cmpl	$9, -128(%rbp)
	jbe	.L73
	movq	$33, -88(%rbp)
	jmp	.L53
.L73:
	movq	$34, -88(%rbp)
	jmp	.L53
.L17:
	cmpl	$5, -132(%rbp)
	je	.L75
	movq	$69, -88(%rbp)
	jmp	.L53
.L75:
	movq	$28, -88(%rbp)
	jmp	.L53
.L18:
	movl	$5, -132(%rbp)
	movq	$51, -88(%rbp)
	jmp	.L53
.L48:
	movl	$0, -104(%rbp)
	movq	$35, -88(%rbp)
	jmp	.L53
.L36:
	movl	-100(%rbp), %eax
	cltq
	movzbl	-18(%rbp,%rax), %eax
	cmpb	$57, %al
	jle	.L77
	movq	$59, -88(%rbp)
	jmp	.L53
.L77:
	movq	$51, -88(%rbp)
	jmp	.L53
.L16:
	cmpl	$9, -124(%rbp)
	jbe	.L79
	movq	$12, -88(%rbp)
	jmp	.L53
.L79:
	movq	$68, -88(%rbp)
	jmp	.L53
.L19:
	movl	-120(%rbp), %eax
	movb	$0, -50(%rbp,%rax)
	addl	$1, -120(%rbp)
	movq	$21, -88(%rbp)
	jmp	.L53
.L31:
	movl	-128(%rbp), %eax
	movb	$0, -80(%rbp,%rax)
	addl	$1, -128(%rbp)
	movq	$40, -88(%rbp)
	jmp	.L53
.L10:
	movl	$3, -132(%rbp)
	movq	$28, -88(%rbp)
	jmp	.L53
.L35:
	leaq	-80(%rbp), %rcx
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -132(%rbp)
	addl	$1, -108(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L53
.L13:
	movl	$5, -132(%rbp)
	movq	$6, -88(%rbp)
	jmp	.L53
.L24:
	addl	$1, -104(%rbp)
	movq	$35, -88(%rbp)
	jmp	.L53
.L25:
	leaq	-136(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -108(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L53
.L49:
	movl	-104(%rbp), %eax
	cltq
	movzbl	-18(%rbp,%rax), %eax
	cmpb	$90, %al
	jle	.L81
	movq	$18, -88(%rbp)
	jmp	.L53
.L81:
	movq	$47, -88(%rbp)
	jmp	.L53
.L32:
	movb	$84, -70(%rbp)
	movb	$85, -69(%rbp)
	movb	$69, -68(%rbp)
	movb	$83, -67(%rbp)
	movb	$68, -66(%rbp)
	movb	$65, -65(%rbp)
	movb	$89, -64(%rbp)
	movb	$0, -63(%rbp)
	movl	$8, -124(%rbp)
	movq	$61, -88(%rbp)
	jmp	.L53
.L26:
	cmpq	$8, -96(%rbp)
	je	.L83
	movq	$77, -88(%rbp)
	jmp	.L53
.L83:
	movq	$36, -88(%rbp)
	jmp	.L53
.L28:
	cmpl	$9, -116(%rbp)
	jbe	.L85
	movq	$31, -88(%rbp)
	jmp	.L53
.L85:
	movq	$25, -88(%rbp)
	jmp	.L53
.L47:
	cmpl	$7, -100(%rbp)
	jg	.L87
	movq	$27, -88(%rbp)
	jmp	.L53
.L87:
	movq	$60, -88(%rbp)
	jmp	.L53
.L30:
	cmpl	$2, -104(%rbp)
	jg	.L89
	movq	$11, -88(%rbp)
	jmp	.L53
.L89:
	movq	$9, -88(%rbp)
	jmp	.L53
.L94:
	nop
.L53:
	jmp	.L91
.L93:
	call	__stack_chk_fail@PLT
.L92:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
