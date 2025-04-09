	.file	"adityas27_CP_Solution_8_flatten.c"
	.text
	.globl	_TIG_IZ_Kknz_argc
	.bss
	.align 4
	.type	_TIG_IZ_Kknz_argc, @object
	.size	_TIG_IZ_Kknz_argc, 4
_TIG_IZ_Kknz_argc:
	.zero	4
	.globl	_TIG_IZ_Kknz_envp
	.align 8
	.type	_TIG_IZ_Kknz_envp, @object
	.size	_TIG_IZ_Kknz_envp, 8
_TIG_IZ_Kknz_envp:
	.zero	8
	.globl	_TIG_IZ_Kknz_argv
	.align 8
	.type	_TIG_IZ_Kknz_argv, @object
	.size	_TIG_IZ_Kknz_argv, 8
_TIG_IZ_Kknz_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"The product of the first %d natural numbers is: %d\n"
.LC1:
	.string	"Enter a positive integer: "
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"Please enter a positive integer."
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Kknz_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Kknz_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Kknz_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 87 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Kknz--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Kknz_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Kknz_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Kknz_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L24:
	cmpq	$12, -16(%rbp)
	ja	.L27
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L27-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L27-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L27-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-28(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L18
	movq	$9, -16(%rbp)
	jmp	.L20
.L18:
	movq	$1, -16(%rbp)
	jmp	.L20
.L7:
	movl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L20
.L11:
	movl	-28(%rbp), %eax
	testl	%eax, %eax
	jg	.L21
	movq	$7, -16(%rbp)
	jmp	.L20
.L21:
	movq	$12, -16(%rbp)
	jmp	.L20
.L16:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L20
.L15:
	movl	$0, %eax
	jmp	.L25
.L10:
	movl	-24(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	addl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L20
.L13:
	movq	$0, -16(%rbp)
	jmp	.L20
.L9:
	movl	$1, %eax
	jmp	.L25
.L17:
	movl	$1, -24(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L20
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -16(%rbp)
	jmp	.L20
.L27:
	nop
.L20:
	jmp	.L24
.L25:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	call	__stack_chk_fail@PLT
.L26:
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
