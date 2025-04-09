	.file	"j0z3ph_2024-2_main4_flatten.c"
	.text
	.globl	_TIG_IZ_pkYh_argv
	.bss
	.align 8
	.type	_TIG_IZ_pkYh_argv, @object
	.size	_TIG_IZ_pkYh_argv, 8
_TIG_IZ_pkYh_argv:
	.zero	8
	.globl	_TIG_IZ_pkYh_envp
	.align 8
	.type	_TIG_IZ_pkYh_envp, @object
	.size	_TIG_IZ_pkYh_envp, 8
_TIG_IZ_pkYh_envp:
	.zero	8
	.globl	_TIG_IZ_pkYh_argc
	.align 4
	.type	_TIG_IZ_pkYh_argc, @object
	.size	_TIG_IZ_pkYh_argc, 4
_TIG_IZ_pkYh_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Hola"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_pkYh_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_pkYh_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_pkYh_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 121 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-pkYh--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_pkYh_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_pkYh_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_pkYh_envp(%rip)
	nop
	movq	$5, -8(%rbp)
.L14:
	cmpq	$5, -8(%rbp)
	je	.L6
	cmpq	$5, -8(%rbp)
	ja	.L16
	cmpq	$3, -8(%rbp)
	je	.L8
	cmpq	$3, -8(%rbp)
	ja	.L16
	cmpq	$0, -8(%rbp)
	je	.L9
	cmpq	$1, -8(%rbp)
	jne	.L16
	sall	-12(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L10
.L8:
	movl	$0, %eax
	jmp	.L15
.L6:
	movl	$100, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L10
.L9:
	cmpl	$9, -12(%rbp)
	jg	.L12
	movq	$1, -8(%rbp)
	jmp	.L10
.L12:
	movq	$3, -8(%rbp)
	jmp	.L10
.L16:
	nop
.L10:
	jmp	.L14
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
