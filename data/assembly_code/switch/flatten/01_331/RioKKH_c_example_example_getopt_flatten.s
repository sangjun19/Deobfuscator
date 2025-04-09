	.file	"RioKKH_c_example_example_getopt_flatten.c"
	.text
	.globl	_TIG_IZ_YPi6_argc
	.bss
	.align 4
	.type	_TIG_IZ_YPi6_argc, @object
	.size	_TIG_IZ_YPi6_argc, 4
_TIG_IZ_YPi6_argc:
	.zero	4
	.globl	_TIG_IZ_YPi6_argv
	.align 8
	.type	_TIG_IZ_YPi6_argv, @object
	.size	_TIG_IZ_YPi6_argv, 8
_TIG_IZ_YPi6_argv:
	.zero	8
	.globl	_TIG_IZ_YPi6_envp
	.align 8
	.type	_TIG_IZ_YPi6_envp, @object
	.size	_TIG_IZ_YPi6_envp, 8
_TIG_IZ_YPi6_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Usage: %s [-a] [-b value]\n"
.LC1:
	.string	"Option b with value '%s'\n"
	.align 8
.LC2:
	.string	"Iteration %d.\n Calling getopt makes optind %d\n"
.LC3:
	.string	"Option a"
.LC4:
	.string	"optind is now %d\n"
.LC5:
	.string	"Non-option argument: %s\n"
	.align 8
.LC6:
	.string	"Initial optind is %d, argc is %d.\n"
.LC7:
	.string	"ab:"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_YPi6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_YPi6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_YPi6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YPi6--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_YPi6_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_YPi6_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_YPi6_envp(%rip)
	nop
	movq	$22, -8(%rbp)
.L31:
	cmpq	$22, -8(%rbp)
	ja	.L32
	movq	-8(%rbp), %rax
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L32-.L8
	.long	.L19-.L8
	.long	.L32-.L8
	.long	.L18-.L8
	.long	.L32-.L8
	.long	.L17-.L8
	.long	.L32-.L8
	.long	.L16-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L32-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L32-.L8
	.long	.L7-.L8
	.text
.L11:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$5, -8(%rbp)
	jmp	.L22
.L14:
	movl	-16(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.L23
	movq	$19, -8(%rbp)
	jmp	.L22
.L23:
	movq	$20, -8(%rbp)
	jmp	.L22
.L13:
	movq	optarg(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L22
.L20:
	cmpl	$-1, -24(%rbp)
	je	.L25
	movq	$3, -8(%rbp)
	jmp	.L22
.L25:
	movq	$13, -8(%rbp)
	jmp	.L22
.L19:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -20(%rbp)
	movl	optind(%rip), %edx
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L22
.L16:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L22
.L15:
	movl	optind(%rip), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	optind(%rip), %eax
	movl	%eax, -16(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L22
.L10:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -16(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L22
.L12:
	movl	$0, -20(%rbp)
	movl	optind(%rip), %eax
	movl	-36(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L22
.L7:
	movq	$17, -8(%rbp)
	jmp	.L22
.L18:
	movl	$1, %eax
	jmp	.L27
.L21:
	movq	-48(%rbp), %rcx
	movl	-36(%rbp), %eax
	leaq	.LC7(%rip), %rdx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	getopt@PLT
	movl	%eax, -24(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L22
.L17:
	cmpl	$97, -24(%rbp)
	je	.L28
	cmpl	$98, -24(%rbp)
	jne	.L29
	movq	$15, -8(%rbp)
	jmp	.L30
.L28:
	movq	$9, -8(%rbp)
	jmp	.L30
.L29:
	movq	$18, -8(%rbp)
	nop
.L30:
	jmp	.L22
.L9:
	movl	$0, %eax
	jmp	.L27
.L32:
	nop
.L22:
	jmp	.L31
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
