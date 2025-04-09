	.file	"Hemant-k20_ACC45DAYSOFCODE-2024_Day-30TRUESCORE_flatten.c"
	.text
	.globl	_TIG_IZ_8JkA_envp
	.bss
	.align 8
	.type	_TIG_IZ_8JkA_envp, @object
	.size	_TIG_IZ_8JkA_envp, 8
_TIG_IZ_8JkA_envp:
	.zero	8
	.globl	_TIG_IZ_8JkA_argc
	.align 4
	.type	_TIG_IZ_8JkA_argc, @object
	.size	_TIG_IZ_8JkA_argc, 4
_TIG_IZ_8JkA_argc:
	.zero	4
	.globl	_TIG_IZ_8JkA_argv
	.align 8
	.type	_TIG_IZ_8JkA_argv, @object
	.size	_TIG_IZ_8JkA_argv, 8
_TIG_IZ_8JkA_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"IMPOSSIBLE"
.LC1:
	.string	"POSSIBLE"
.LC2:
	.string	"%d"
.LC3:
	.string	"%d%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_8JkA_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8JkA_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8JkA_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8JkA--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_8JkA_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_8JkA_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_8JkA_envp(%rip)
	nop
	movq	$8, -16(%rbp)
.L27:
	cmpq	$13, -16(%rbp)
	ja	.L30
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L30-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L28
	jmp	.L29
.L9:
	movl	-28(%rbp), %edx
	movl	-36(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L20
	movq	$10, -16(%rbp)
	jmp	.L22
.L20:
	movq	$9, -16(%rbp)
	jmp	.L22
.L12:
	movq	$7, -16(%rbp)
	jmp	.L22
.L17:
	addl	$1, -20(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L22
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L7:
	movl	-40(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L23
	movq	$2, -16(%rbp)
	jmp	.L22
.L23:
	movq	$4, -16(%rbp)
	jmp	.L22
.L10:
	movl	-24(%rbp), %edx
	movl	-32(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L25
	movq	$0, -16(%rbp)
	jmp	.L22
.L25:
	movq	$3, -16(%rbp)
	jmp	.L22
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L22
.L13:
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -20(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L22
.L16:
	leaq	-32(%rbp), %rdx
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L22
.L30:
	nop
.L22:
	jmp	.L27
.L29:
	call	__stack_chk_fail@PLT
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
