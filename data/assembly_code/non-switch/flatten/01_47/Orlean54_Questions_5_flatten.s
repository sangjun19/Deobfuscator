	.file	"Orlean54_Questions_5_flatten.c"
	.text
	.globl	_TIG_IZ_YabY_argv
	.bss
	.align 8
	.type	_TIG_IZ_YabY_argv, @object
	.size	_TIG_IZ_YabY_argv, 8
_TIG_IZ_YabY_argv:
	.zero	8
	.globl	_TIG_IZ_YabY_argc
	.align 4
	.type	_TIG_IZ_YabY_argc, @object
	.size	_TIG_IZ_YabY_argc, 4
_TIG_IZ_YabY_argc:
	.zero	4
	.globl	_TIG_IZ_YabY_envp
	.align 8
	.type	_TIG_IZ_YabY_envp, @object
	.size	_TIG_IZ_YabY_envp, 8
_TIG_IZ_YabY_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Number: %d\n"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_YabY_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_YabY_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_YabY_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YabY--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_YabY_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_YabY_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_YabY_envp(%rip)
	nop
	movq	$5, -24(%rbp)
.L14:
	cmpq	$6, -24(%rbp)
	je	.L6
	cmpq	$6, -24(%rbp)
	ja	.L17
	cmpq	$5, -24(%rbp)
	je	.L8
	cmpq	$5, -24(%rbp)
	ja	.L17
	cmpq	$2, -24(%rbp)
	je	.L9
	cmpq	$3, -24(%rbp)
	jne	.L17
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L15
	jmp	.L16
.L6:
	leaq	-28(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-28(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L11
.L8:
	movl	$0, -28(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L11
.L9:
	movl	-28(%rbp), %eax
	cmpl	$5, %eax
	jg	.L12
	movq	$6, -24(%rbp)
	jmp	.L11
.L12:
	movq	$3, -24(%rbp)
	jmp	.L11
.L17:
	nop
.L11:
	jmp	.L14
.L16:
	call	__stack_chk_fail@PLT
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
