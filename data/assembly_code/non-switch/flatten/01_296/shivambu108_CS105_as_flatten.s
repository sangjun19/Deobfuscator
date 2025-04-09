	.file	"shivambu108_CS105_as_flatten.c"
	.text
	.globl	_TIG_IZ_lsrd_argv
	.bss
	.align 8
	.type	_TIG_IZ_lsrd_argv, @object
	.size	_TIG_IZ_lsrd_argv, 8
_TIG_IZ_lsrd_argv:
	.zero	8
	.globl	_TIG_IZ_lsrd_envp
	.align 8
	.type	_TIG_IZ_lsrd_envp, @object
	.size	_TIG_IZ_lsrd_envp, 8
_TIG_IZ_lsrd_envp:
	.zero	8
	.globl	_TIG_IZ_lsrd_argc
	.align 4
	.type	_TIG_IZ_lsrd_argc, @object
	.size	_TIG_IZ_lsrd_argc, 4
_TIG_IZ_lsrd_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%1d %2d %3d"
.LC1:
	.string	"Sum=%d"
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
	movq	$0, _TIG_IZ_lsrd_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_lsrd_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_lsrd_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-lsrd--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_lsrd_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_lsrd_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_lsrd_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L11:
	cmpq	$2, -16(%rbp)
	je	.L6
	cmpq	$2, -16(%rbp)
	ja	.L14
	cmpq	$0, -16(%rbp)
	je	.L8
	cmpq	$1, -16(%rbp)
	jne	.L14
	leaq	-20(%rbp), %rcx
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	movq	$1, -16(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
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
