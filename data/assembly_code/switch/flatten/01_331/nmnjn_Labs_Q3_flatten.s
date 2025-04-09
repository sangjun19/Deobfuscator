	.file	"nmnjn_Labs_Q3_flatten.c"
	.text
	.globl	_TIG_IZ_hZj0_argc
	.bss
	.align 4
	.type	_TIG_IZ_hZj0_argc, @object
	.size	_TIG_IZ_hZj0_argc, 4
_TIG_IZ_hZj0_argc:
	.zero	4
	.globl	_TIG_IZ_hZj0_envp
	.align 8
	.type	_TIG_IZ_hZj0_envp, @object
	.size	_TIG_IZ_hZj0_envp, 8
_TIG_IZ_hZj0_envp:
	.zero	8
	.globl	_TIG_IZ_hZj0_argv
	.align 8
	.type	_TIG_IZ_hZj0_argv, @object
	.size	_TIG_IZ_hZj0_argv, 8
_TIG_IZ_hZj0_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Inside Child"
.LC1:
	.string	"The child pid is: %d\n"
.LC2:
	.string	"The parent pid is: %d\n"
.LC3:
	.string	"The process id is: %d\n"
.LC4:
	.string	"Inside Parent"
.LC5:
	.string	"Fork Failed\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_hZj0_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_hZj0_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_hZj0_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-hZj0--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_hZj0_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_hZj0_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_hZj0_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L19:
	cmpq	$8, -16(%rbp)
	ja	.L22
	movq	-16(%rbp), %rax
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
	.long	.L22-.L8
	.long	.L22-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L22-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	getpid@PLT
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getppid@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movl	$1, %edi
	call	exit@PLT
.L12:
	call	fork@PLT
	movl	%eax, -36(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L14
.L10:
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	wait@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	getpid@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getppid@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movq	$5, -16(%rbp)
	jmp	.L14
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	jmp	.L21
.L9:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L13:
	cmpl	$-1, -36(%rbp)
	je	.L16
	cmpl	$0, -36(%rbp)
	jne	.L17
	movq	$8, -16(%rbp)
	jmp	.L18
.L16:
	movq	$7, -16(%rbp)
	jmp	.L18
.L17:
	movq	$6, -16(%rbp)
	nop
.L18:
	jmp	.L14
.L22:
	nop
.L14:
	jmp	.L19
.L21:
	call	__stack_chk_fail@PLT
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
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
