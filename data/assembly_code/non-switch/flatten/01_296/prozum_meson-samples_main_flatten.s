	.file	"prozum_meson-samples_main_flatten.c"
	.text
	.globl	_TIG_IZ_ctU2_argc
	.bss
	.align 4
	.type	_TIG_IZ_ctU2_argc, @object
	.size	_TIG_IZ_ctU2_argc, 4
_TIG_IZ_ctU2_argc:
	.zero	4
	.globl	_TIG_IZ_ctU2_envp
	.align 8
	.type	_TIG_IZ_ctU2_envp, @object
	.size	_TIG_IZ_ctU2_envp, 8
_TIG_IZ_ctU2_envp:
	.zero	8
	.globl	_TIG_IZ_ctU2_argv
	.align 8
	.type	_TIG_IZ_ctU2_argv, @object
	.size	_TIG_IZ_ctU2_argv, 8
_TIG_IZ_ctU2_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Super advanced example!"
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
	movq	$0, _TIG_IZ_ctU2_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ctU2_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ctU2_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ctU2--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_ctU2_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_ctU2_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_ctU2_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L10:
	cmpq	$0, -8(%rbp)
	je	.L6
	cmpq	$1, -8(%rbp)
	jne	.L12
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L8
.L6:
	movl	$0, %eax
	jmp	.L11
.L12:
	nop
.L8:
	jmp	.L10
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
