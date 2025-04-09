	.file	"sabboshachi_C-Cpp_Practice_code05_flatten.c"
	.text
	.globl	_TIG_IZ_Ro5X_argc
	.bss
	.align 4
	.type	_TIG_IZ_Ro5X_argc, @object
	.size	_TIG_IZ_Ro5X_argc, 4
_TIG_IZ_Ro5X_argc:
	.zero	4
	.globl	_TIG_IZ_Ro5X_envp
	.align 8
	.type	_TIG_IZ_Ro5X_envp, @object
	.size	_TIG_IZ_Ro5X_envp, 8
_TIG_IZ_Ro5X_envp:
	.zero	8
	.globl	_TIG_IZ_Ro5X_argv
	.align 8
	.type	_TIG_IZ_Ro5X_argv, @object
	.size	_TIG_IZ_Ro5X_argv, 8
_TIG_IZ_Ro5X_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"x is 3"
.LC1:
	.string	"x is 2"
.LC2:
	.string	"x is 1"
	.align 8
.LC3:
	.string	"x has a value other than 1 2 3"
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
	movq	$0, _TIG_IZ_Ro5X_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Ro5X_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Ro5X_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Ro5X--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Ro5X_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Ro5X_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Ro5X_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L23:
	cmpq	$11, -8(%rbp)
	ja	.L25
	movq	-8(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L25-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L25-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	movq	$0, -8(%rbp)
	jmp	.L16
.L14:
	movl	$0, %eax
	jmp	.L24
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L15:
	movl	$2, -16(%rbp)
	movl	$5, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L16
.L10:
	cmpl	$21, -16(%rbp)
	je	.L18
	cmpl	$21, -16(%rbp)
	jg	.L19
	cmpl	$1, -16(%rbp)
	je	.L20
	cmpl	$2, -16(%rbp)
	je	.L21
	jmp	.L19
.L18:
	movq	$3, -8(%rbp)
	jmp	.L22
.L20:
	movq	$11, -8(%rbp)
	jmp	.L22
.L21:
	movq	$6, -8(%rbp)
	jmp	.L22
.L19:
	movq	$10, -8(%rbp)
	nop
.L22:
	jmp	.L16
.L25:
	nop
.L16:
	jmp	.L23
.L24:
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
