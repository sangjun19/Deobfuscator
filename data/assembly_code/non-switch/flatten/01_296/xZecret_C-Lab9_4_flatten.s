	.file	"xZecret_C-Lab9_4_flatten.c"
	.text
	.globl	_TIG_IZ_uWPl_argc
	.bss
	.align 4
	.type	_TIG_IZ_uWPl_argc, @object
	.size	_TIG_IZ_uWPl_argc, 4
_TIG_IZ_uWPl_argc:
	.zero	4
	.globl	_TIG_IZ_uWPl_argv
	.align 8
	.type	_TIG_IZ_uWPl_argv, @object
	.size	_TIG_IZ_uWPl_argv, 8
_TIG_IZ_uWPl_argv:
	.zero	8
	.globl	_TIG_IZ_uWPl_envp
	.align 8
	.type	_TIG_IZ_uWPl_envp, @object
	.size	_TIG_IZ_uWPl_envp, 8
_TIG_IZ_uWPl_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"piboqrboi'8 bt7t8635 b1tq\n\t qr ilvq 91 qvb35 q"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_uWPl_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_uWPl_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_uWPl_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-uWPl--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_uWPl_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_uWPl_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_uWPl_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L15:
	cmpq	$5, -16(%rbp)
	je	.L6
	cmpq	$5, -16(%rbp)
	ja	.L17
	cmpq	$4, -16(%rbp)
	je	.L8
	cmpq	$4, -16(%rbp)
	ja	.L17
	cmpq	$0, -16(%rbp)
	je	.L9
	cmpq	$3, -16(%rbp)
	je	.L10
	jmp	.L17
.L8:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -24(%rbp)
	movq	-8(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	check_type
	movq	$3, -16(%rbp)
	jmp	.L11
.L10:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L12
	movq	$4, -16(%rbp)
	jmp	.L11
.L12:
	movq	$0, -16(%rbp)
	jmp	.L11
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L11
.L9:
	movl	$0, %eax
	jmp	.L16
.L17:
	nop
.L11:
	jmp	.L15
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
.LC1:
	.string	"isalpha "
.LC2:
	.string	"iscntrl "
.LC3:
	.string	"isupper "
.LC4:
	.string	"isxdigit "
.LC5:
	.string	"is >> "
.LC6:
	.string	"isprint "
.LC7:
	.string	"ispunct "
.LC8:
	.string	"isdigit "
.LC9:
	.string	"isspace "
.LC10:
	.string	"islower "
.LC11:
	.string	"isalnum "
	.text
	.globl	check_type
	.type	check_type, @function
check_type:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -100(%rbp)
	movq	$19, -8(%rbp)
.L76:
	cmpq	$32, -8(%rbp)
	ja	.L77
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L53-.L21
	.long	.L52-.L21
	.long	.L51-.L21
	.long	.L50-.L21
	.long	.L49-.L21
	.long	.L48-.L21
	.long	.L47-.L21
	.long	.L46-.L21
	.long	.L45-.L21
	.long	.L44-.L21
	.long	.L78-.L21
	.long	.L42-.L21
	.long	.L41-.L21
	.long	.L40-.L21
	.long	.L39-.L21
	.long	.L38-.L21
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L33-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L35:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$27, -8(%rbp)
	jmp	.L54
.L28:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -8(%rbp)
	jmp	.L54
.L49:
	movl	$10, %edi
	call	putchar@PLT
	movq	$10, -8(%rbp)
	jmp	.L54
.L23:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L54
.L39:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L54
.L38:
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2, %eax
	testl	%eax, %eax
	je	.L55
	movq	$25, -8(%rbp)
	jmp	.L54
.L55:
	movq	$31, -8(%rbp)
	jmp	.L54
.L22:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L54
.L41:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L57
	movq	$0, -8(%rbp)
	jmp	.L54
.L57:
	movq	$21, -8(%rbp)
	jmp	.L54
.L45:
	movq	-88(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L59
	movq	$18, -8(%rbp)
	jmp	.L54
.L59:
	movq	$27, -8(%rbp)
	jmp	.L54
.L52:
	movq	-72(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$4096, %eax
	testl	%eax, %eax
	je	.L61
	movq	$11, -8(%rbp)
	jmp	.L54
.L61:
	movq	$7, -8(%rbp)
	jmp	.L54
.L30:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$4, %eax
	testl	%eax, %eax
	je	.L63
	movq	$32, -8(%rbp)
	jmp	.L54
.L63:
	movq	$5, -8(%rbp)
	jmp	.L54
.L50:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L54
.L37:
	call	__ctype_b_loc@PLT
	movq	%rax, -72(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L54
.L29:
	movq	-64(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8, %eax
	testl	%eax, %eax
	je	.L65
	movq	$20, -8(%rbp)
	jmp	.L54
.L65:
	movq	$22, -8(%rbp)
	jmp	.L54
.L32:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L54
.L27:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$256, %eax
	testl	%eax, %eax
	je	.L67
	movq	$3, -8(%rbp)
	jmp	.L54
.L67:
	movq	$14, -8(%rbp)
	jmp	.L54
.L42:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L54
.L44:
	movl	-100(%rbp), %eax
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	__ctype_b_loc@PLT
	movq	%rax, -88(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L54
.L40:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$30, -8(%rbp)
	jmp	.L54
.L34:
	movq	$9, -8(%rbp)
	jmp	.L54
.L20:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L54
.L36:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$512, %eax
	testl	%eax, %eax
	je	.L69
	movq	$2, -8(%rbp)
	jmp	.L54
.L69:
	movq	$4, -8(%rbp)
	jmp	.L54
.L47:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L54
.L26:
	call	__ctype_b_loc@PLT
	movq	%rax, -80(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L54
.L31:
	call	__ctype_b_loc@PLT
	movq	%rax, -56(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L54
.L25:
	movq	-80(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L71
	movq	$6, -8(%rbp)
	jmp	.L54
.L71:
	movq	$16, -8(%rbp)
	jmp	.L54
.L48:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L54
.L53:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -8(%rbp)
	jmp	.L54
.L46:
	call	__ctype_b_loc@PLT
	movq	%rax, -64(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L54
.L24:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$16384, %eax
	testl	%eax, %eax
	je	.L74
	movq	$13, -8(%rbp)
	jmp	.L54
.L74:
	movq	$30, -8(%rbp)
	jmp	.L54
.L51:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L54
.L33:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -8(%rbp)
	jmp	.L54
.L77:
	nop
.L54:
	jmp	.L76
.L78:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	check_type, .-check_type
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
