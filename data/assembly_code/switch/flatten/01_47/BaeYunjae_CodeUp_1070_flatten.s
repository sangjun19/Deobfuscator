	.file	"BaeYunjae_CodeUp_1070_flatten.c"
	.text
	.globl	_TIG_IZ_Ie6P_argv
	.bss
	.align 8
	.type	_TIG_IZ_Ie6P_argv, @object
	.size	_TIG_IZ_Ie6P_argv, 8
_TIG_IZ_Ie6P_argv:
	.zero	8
	.globl	_TIG_IZ_Ie6P_envp
	.align 8
	.type	_TIG_IZ_Ie6P_envp, @object
	.size	_TIG_IZ_Ie6P_envp, 8
_TIG_IZ_Ie6P_envp:
	.zero	8
	.globl	_TIG_IZ_Ie6P_argc
	.align 4
	.type	_TIG_IZ_Ie6P_argc, @object
	.size	_TIG_IZ_Ie6P_argc, 4
_TIG_IZ_Ie6P_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"summer"
.LC1:
	.string	"fall"
.LC2:
	.string	"winter"
.LC3:
	.string	"spring"
.LC4:
	.string	"%d"
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
	movq	$0, _TIG_IZ_Ie6P_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Ie6P_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Ie6P_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Ie6P--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Ie6P_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Ie6P_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Ie6P_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L25:
	cmpq	$11, -16(%rbp)
	ja	.L28
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
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L10:
	movq	$2, -16(%rbp)
	jmp	.L16
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L9:
	movl	-20(%rbp), %eax
	cmpl	$12, %eax
	ja	.L17
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L17-.L19
	.long	.L18-.L19
	.long	.L18-.L19
	.long	.L22-.L19
	.long	.L22-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L21-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L20-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L20:
	movq	$1, -16(%rbp)
	jmp	.L23
.L21:
	movq	$4, -16(%rbp)
	jmp	.L23
.L22:
	movq	$11, -16(%rbp)
	jmp	.L23
.L18:
	movq	$3, -16(%rbp)
	jmp	.L23
.L17:
	movq	$8, -16(%rbp)
	nop
.L23:
	jmp	.L16
.L11:
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L16
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L28:
	nop
.L16:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
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
