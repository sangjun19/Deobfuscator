	.file	"ItsKendi_alx-low_level_programming_3-main_flatten.c"
	.text
	.globl	_TIG_IZ_NABd_argv
	.bss
	.align 8
	.type	_TIG_IZ_NABd_argv, @object
	.size	_TIG_IZ_NABd_argv, 8
_TIG_IZ_NABd_argv:
	.zero	8
	.globl	_TIG_IZ_NABd_envp
	.align 8
	.type	_TIG_IZ_NABd_envp, @object
	.size	_TIG_IZ_NABd_envp, 8
_TIG_IZ_NABd_envp:
	.zero	8
	.globl	_TIG_IZ_NABd_argc
	.align 4
	.type	_TIG_IZ_NABd_argc, @object
	.size	_TIG_IZ_NABd_argc, 4
_TIG_IZ_NABd_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Address of variable 'p'; %p\n"
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
	movq	$0, _TIG_IZ_NABd_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_NABd_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_NABd_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NABd--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_NABd_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_NABd_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_NABd_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L10:
	cmpq	$0, -16(%rbp)
	je	.L6
	cmpq	$1, -16(%rbp)
	jne	.L13
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L8
.L6:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L11
	jmp	.L12
.L13:
	nop
.L8:
	jmp	.L10
.L12:
	call	__stack_chk_fail@PLT
.L11:
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
