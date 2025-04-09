	.file	"azaregon_Fundamental-Programming-code_05_magic_herb_flatten.c"
	.text
	.globl	_TIG_IZ_60kl_argv
	.bss
	.align 8
	.type	_TIG_IZ_60kl_argv, @object
	.size	_TIG_IZ_60kl_argv, 8
_TIG_IZ_60kl_argv:
	.zero	8
	.globl	_TIG_IZ_60kl_argc
	.align 4
	.type	_TIG_IZ_60kl_argc, @object
	.size	_TIG_IZ_60kl_argc, 4
_TIG_IZ_60kl_argc:
	.zero	4
	.globl	_TIG_IZ_60kl_envp
	.align 8
	.type	_TIG_IZ_60kl_envp, @object
	.size	_TIG_IZ_60kl_envp, 8
_TIG_IZ_60kl_envp:
	.zero	8
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
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_60kl_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_60kl_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_60kl_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-60kl--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_60kl_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_60kl_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_60kl_envp(%rip)
	nop
	movq	$1, -152(%rbp)
.L17:
	cmpq	$6, -152(%rbp)
	ja	.L20
	movq	-152(%rbp), %rax
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
	.long	.L9-.L8
	.long	.L20-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$94, -144(%rbp)
	movl	$222, -140(%rbp)
	movl	$221, -136(%rbp)
	movl	$186, -132(%rbp)
	movl	$181, -128(%rbp)
	movl	$208, -124(%rbp)
	movl	$207, -120(%rbp)
	movl	$110, -116(%rbp)
	movl	$187, -112(%rbp)
	movl	$185, -108(%rbp)
	movl	$17, -104(%rbp)
	movl	$212, -100(%rbp)
	movl	$115, -96(%rbp)
	movl	$215, -92(%rbp)
	movl	$100, -88(%rbp)
	movl	$94, -80(%rbp)
	movl	$222, -76(%rbp)
	movl	$221, -72(%rbp)
	movl	$186, -68(%rbp)
	movl	$181, -64(%rbp)
	movl	$208, -60(%rbp)
	movl	$207, -56(%rbp)
	movl	$110, -52(%rbp)
	movl	$187, -48(%rbp)
	movl	$185, -44(%rbp)
	movl	$17, -40(%rbp)
	movl	$212, -36(%rbp)
	movl	$115, -32(%rbp)
	movl	$215, -28(%rbp)
	movl	$100, -24(%rbp)
	movl	$15, -160(%rbp)
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	movl	$15, %edi
	call	dechiper
	movl	$0, -156(%rbp)
	movq	$3, -152(%rbp)
	jmp	.L13
.L11:
	movq	$4, -152(%rbp)
	jmp	.L13
.L10:
	movl	-160(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -156(%rbp)
	jge	.L14
	movq	$6, -152(%rbp)
	jmp	.L13
.L14:
	movq	$0, -152(%rbp)
	jmp	.L13
.L7:
	movl	-156(%rbp), %eax
	cltq
	movl	-80(%rbp,%rax,4), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -156(%rbp)
	movq	$3, -152(%rbp)
	jmp	.L13
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
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
	.globl	dechiper
	.type	dechiper, @function
dechiper:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$4, -8(%rbp)
.L61:
	cmpq	$20, -8(%rbp)
	ja	.L62
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
	.long	.L62-.L24
	.long	.L62-.L24
	.long	.L40-.L24
	.long	.L39-.L24
	.long	.L38-.L24
	.long	.L37-.L24
	.long	.L36-.L24
	.long	.L35-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L62-.L24
	.long	.L62-.L24
	.long	.L23-.L24
	.text
.L38:
	cmpl	$0, -36(%rbp)
	jne	.L41
	movq	$5, -8(%rbp)
	jmp	.L43
.L41:
	movq	$11, -8(%rbp)
	jmp	.L43
.L28:
	cmpl	$64, -32(%rbp)
	jle	.L44
	movq	$15, -8(%rbp)
	jmp	.L43
.L44:
	movq	$9, -8(%rbp)
	jmp	.L43
.L27:
	cmpl	$122, -32(%rbp)
	jg	.L46
	movq	$3, -8(%rbp)
	jmp	.L43
.L46:
	movq	$9, -8(%rbp)
	jmp	.L43
.L30:
	movl	$0, %eax
	jmp	.L48
.L34:
	cmpl	$64, -20(%rbp)
	jle	.L49
	movq	$13, -8(%rbp)
	jmp	.L43
.L49:
	movq	$20, -8(%rbp)
	jmp	.L43
.L39:
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-32(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$20, -8(%rbp)
	jmp	.L43
.L26:
	cmpl	$122, -24(%rbp)
	jg	.L51
	movq	$17, -8(%rbp)
	jmp	.L43
.L51:
	movq	$8, -8(%rbp)
	jmp	.L43
.L31:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -32(%rbp)
	movl	-12(%rbp), %eax
	subl	-16(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-12(%rbp), %eax
	addl	%eax, %eax
	subl	-16(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-12(%rbp), %eax
	subl	$3, %eax
	leal	(%rax,%rax), %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L43
.L33:
	cmpl	$64, -28(%rbp)
	jle	.L53
	movq	$7, -8(%rbp)
	jmp	.L43
.L53:
	movq	$6, -8(%rbp)
	jmp	.L43
.L29:
	cmpl	$122, -20(%rbp)
	jg	.L55
	movq	$2, -8(%rbp)
	jmp	.L43
.L55:
	movq	$20, -8(%rbp)
	jmp	.L43
.L25:
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-24(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$20, -8(%rbp)
	jmp	.L43
.L36:
	cmpl	$64, -24(%rbp)
	jle	.L57
	movq	$16, -8(%rbp)
	jmp	.L43
.L57:
	movq	$8, -8(%rbp)
	jmp	.L43
.L37:
	movl	$0, %eax
	jmp	.L48
.L32:
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$20, -8(%rbp)
	jmp	.L43
.L35:
	cmpl	$122, -28(%rbp)
	jg	.L59
	movq	$10, -8(%rbp)
	jmp	.L43
.L59:
	movq	$6, -8(%rbp)
	jmp	.L43
.L40:
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$20, -8(%rbp)
	jmp	.L43
.L23:
	movl	-36(%rbp), %eax
	leal	-1(%rax), %edx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movl	%edx, %edi
	call	dechiper
	movq	$12, -8(%rbp)
	jmp	.L43
.L62:
	nop
.L43:
	jmp	.L61
.L48:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	dechiper, .-dechiper
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
