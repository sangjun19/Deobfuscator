	.file	"Programacion-en-c-moderno_Programacion-C-Clase3_main_flatten.c"
	.text
	.globl	_TIG_IZ_kJh9_envp
	.bss
	.align 8
	.type	_TIG_IZ_kJh9_envp, @object
	.size	_TIG_IZ_kJh9_envp, 8
_TIG_IZ_kJh9_envp:
	.zero	8
	.globl	_TIG_IZ_kJh9_argc
	.align 4
	.type	_TIG_IZ_kJh9_argc, @object
	.size	_TIG_IZ_kJh9_argc, 4
_TIG_IZ_kJh9_argc:
	.zero	4
	.globl	_TIG_IZ_kJh9_argv
	.align 8
	.type	_TIG_IZ_kJh9_argv, @object
	.size	_TIG_IZ_kJh9_argv, 8
_TIG_IZ_kJh9_argv:
	.zero	8
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
	movq	$0, _TIG_IZ_kJh9_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_kJh9_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_kJh9_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 121 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-kJh9--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_kJh9_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_kJh9_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_kJh9_envp(%rip)
	nop
	movq	$5, -8(%rbp)
.L14:
	cmpq	$6, -8(%rbp)
	je	.L6
	cmpq	$6, -8(%rbp)
	ja	.L16
	cmpq	$5, -8(%rbp)
	je	.L8
	cmpq	$5, -8(%rbp)
	ja	.L16
	cmpq	$2, -8(%rbp)
	je	.L9
	cmpq	$3, -8(%rbp)
	jne	.L16
	movl	$0, %eax
	jmp	.L15
.L6:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L11
.L8:
	movl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L11
.L9:
	movl	-12(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L12
	movq	$6, -8(%rbp)
	jmp	.L11
.L12:
	movq	$3, -8(%rbp)
	jmp	.L11
.L16:
	nop
.L11:
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
