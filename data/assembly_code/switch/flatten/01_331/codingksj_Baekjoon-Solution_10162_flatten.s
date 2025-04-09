	.file	"codingksj_Baekjoon-Solution_10162_flatten.c"
	.text
	.globl	_TIG_IZ_aB4k_envp
	.bss
	.align 8
	.type	_TIG_IZ_aB4k_envp, @object
	.size	_TIG_IZ_aB4k_envp, 8
_TIG_IZ_aB4k_envp:
	.zero	8
	.globl	_TIG_IZ_aB4k_argv
	.align 8
	.type	_TIG_IZ_aB4k_argv, @object
	.size	_TIG_IZ_aB4k_argv, 8
_TIG_IZ_aB4k_argv:
	.zero	8
	.globl	_TIG_IZ_aB4k_argc
	.align 4
	.type	_TIG_IZ_aB4k_argc, @object
	.size	_TIG_IZ_aB4k_argc, 4
_TIG_IZ_aB4k_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d %d %d\n"
.LC2:
	.string	"-1"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_aB4k_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_aB4k_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_aB4k_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-aB4k--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_aB4k_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_aB4k_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_aB4k_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L18:
	cmpq	$6, -16(%rbp)
	ja	.L21
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
	.long	.L21-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$0, -28(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	movl	$0, -32(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$458129845, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$5, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -28(%rbp)
	movl	-32(%rbp), %edx
	movslq	%edx, %rax
	imulq	$458129845, %rax, %rax
	shrq	$32, %rax
	sarl	$5, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$300, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$5, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -24(%rbp)
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$5, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	imull	$60, %edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	movl	%edx, -32(%rbp)
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -20(%rbp)
	movl	-32(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, -32(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L14
.L13:
	movl	-20(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L7:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L9:
	movq	$4, -16(%rbp)
	jmp	.L14
.L12:
	movl	-32(%rbp), %eax
	testl	%eax, %eax
	jne	.L16
	movq	$1, -16(%rbp)
	jmp	.L17
.L16:
	movq	$6, -16(%rbp)
	nop
.L17:
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
