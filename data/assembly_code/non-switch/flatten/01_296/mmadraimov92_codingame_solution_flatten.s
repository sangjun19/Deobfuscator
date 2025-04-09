	.file	"mmadraimov92_codingame_solution_flatten.c"
	.text
	.globl	_TIG_IZ_1Xgw_argc
	.bss
	.align 4
	.type	_TIG_IZ_1Xgw_argc, @object
	.size	_TIG_IZ_1Xgw_argc, 4
_TIG_IZ_1Xgw_argc:
	.zero	4
	.globl	_TIG_IZ_1Xgw_argv
	.align 8
	.type	_TIG_IZ_1Xgw_argv, @object
	.size	_TIG_IZ_1Xgw_argv, 8
_TIG_IZ_1Xgw_argv:
	.zero	8
	.globl	_TIG_IZ_1Xgw_envp
	.align 8
	.type	_TIG_IZ_1Xgw_envp, @object
	.size	_TIG_IZ_1Xgw_envp, 8
_TIG_IZ_1Xgw_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d%d%d%d%d%d%d"
.LC2:
	.string	"0 0"
.LC3:
	.string	"0 4"
.LC4:
	.string	"0 1"
.LC5:
	.string	"0 2"
.LC6:
	.string	"0 3"
.LC7:
	.string	"%d%d"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1Xgw_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1Xgw_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1Xgw_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 115 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1Xgw--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_1Xgw_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_1Xgw_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_1Xgw_envp(%rip)
	nop
	movq	$8, -16(%rbp)
.L33:
	cmpq	$17, -16(%rbp)
	ja	.L35
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
	.long	.L35-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L35-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	-60(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -20(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L22
.L12:
	movl	-44(%rbp), %eax
	cmpl	$2449, %eax
	jg	.L23
	movq	$13, -16(%rbp)
	jmp	.L22
.L23:
	movq	$16, -16(%rbp)
	jmp	.L22
.L16:
	movq	$14, -16(%rbp)
	jmp	.L22
.L21:
	leaq	-32(%rbp), %r8
	leaq	-36(%rbp), %rdi
	leaq	-40(%rbp), %rcx
	leaq	-44(%rbp), %rdx
	leaq	-48(%rbp), %rax
	leaq	-24(%rbp), %rsi
	pushq	%rsi
	leaq	-28(%rbp), %rsi
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addq	$16, %rsp
	movq	$7, -16(%rbp)
	jmp	.L22
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L15:
	movl	-44(%rbp), %eax
	cmpl	$2299, %eax
	jg	.L25
	movq	$17, -16(%rbp)
	jmp	.L22
.L25:
	movq	$12, -16(%rbp)
	jmp	.L22
.L11:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L7:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L18:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L19:
	leaq	-52(%rbp), %rdx
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -20(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L22
.L14:
	movl	-60(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L27
	movq	$5, -16(%rbp)
	jmp	.L22
.L27:
	movq	$1, -16(%rbp)
	jmp	.L22
.L17:
	movl	-44(%rbp), %eax
	cmpl	$1649, %eax
	jg	.L29
	movq	$11, -16(%rbp)
	jmp	.L22
.L29:
	movq	$2, -16(%rbp)
	jmp	.L22
.L20:
	movl	-44(%rbp), %eax
	cmpl	$2199, %eax
	jg	.L31
	movq	$6, -16(%rbp)
	jmp	.L22
.L31:
	movq	$9, -16(%rbp)
	jmp	.L22
.L35:
	nop
.L22:
	jmp	.L33
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
