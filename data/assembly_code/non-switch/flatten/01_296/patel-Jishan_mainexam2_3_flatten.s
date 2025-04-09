	.file	"patel-Jishan_mainexam2_3_flatten.c"
	.text
	.globl	_TIG_IZ_bSGF_envp
	.bss
	.align 8
	.type	_TIG_IZ_bSGF_envp, @object
	.size	_TIG_IZ_bSGF_envp, 8
_TIG_IZ_bSGF_envp:
	.zero	8
	.globl	_TIG_IZ_bSGF_argv
	.align 8
	.type	_TIG_IZ_bSGF_argv, @object
	.size	_TIG_IZ_bSGF_argv, 8
_TIG_IZ_bSGF_argv:
	.zero	8
	.globl	_TIG_IZ_bSGF_argc
	.align 4
	.type	_TIG_IZ_bSGF_argc, @object
	.size	_TIG_IZ_bSGF_argc, 4
_TIG_IZ_bSGF_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"a=%d:b=%d:"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_bSGF_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_bSGF_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_bSGF_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-bSGF--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_bSGF_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_bSGF_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_bSGF_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L11:
	cmpq	$2, -8(%rbp)
	je	.L6
	cmpq	$2, -8(%rbp)
	ja	.L13
	cmpq	$0, -8(%rbp)
	je	.L8
	cmpq	$1, -8(%rbp)
	jne	.L13
	movl	$7, -16(%rbp)
	movl	$4, -12(%rbp)
	movl	-12(%rbp), %eax
	addl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	subl	-12(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	subl	%eax, -16(%rbp)
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	jmp	.L12
.L6:
	movq	$1, -8(%rbp)
	jmp	.L9
.L13:
	nop
.L9:
	jmp	.L11
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
