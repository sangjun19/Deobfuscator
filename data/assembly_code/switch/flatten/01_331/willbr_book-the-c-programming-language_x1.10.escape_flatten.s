	.file	"willbr_book-the-c-programming-language_x1.10.escape_flatten.c"
	.text
	.globl	_TIG_IZ_D1We_argc
	.bss
	.align 4
	.type	_TIG_IZ_D1We_argc, @object
	.size	_TIG_IZ_D1We_argc, 4
_TIG_IZ_D1We_argc:
	.zero	4
	.globl	_TIG_IZ_D1We_argv
	.align 8
	.type	_TIG_IZ_D1We_argv, @object
	.size	_TIG_IZ_D1We_argv, 8
_TIG_IZ_D1We_argv:
	.zero	8
	.globl	_TIG_IZ_D1We_envp
	.align 8
	.type	_TIG_IZ_D1We_envp, @object
	.size	_TIG_IZ_D1We_envp, 8
_TIG_IZ_D1We_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\\b"
.LC1:
	.string	"\\t"
.LC2:
	.string	"\\\\"
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
	movq	$0, _TIG_IZ_D1We_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_D1We_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_D1We_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 121 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-D1We--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_D1We_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_D1We_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_D1We_envp(%rip)
	nop
	movq	$9, -8(%rbp)
.L26:
	cmpq	$12, -8(%rbp)
	ja	.L28
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$11, -8(%rbp)
	jmp	.L17
.L7:
	movl	$0, %eax
	jmp	.L27
.L9:
	call	getchar@PLT
	movl	%eax, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L17
.L11:
	movq	$11, -8(%rbp)
	jmp	.L17
.L13:
	cmpl	$-1, -12(%rbp)
	je	.L19
	movq	$2, -8(%rbp)
	jmp	.L17
.L19:
	movq	$12, -8(%rbp)
	jmp	.L17
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L17
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L17
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L17
.L15:
	cmpl	$92, -12(%rbp)
	je	.L21
	cmpl	$92, -12(%rbp)
	jg	.L22
	cmpl	$8, -12(%rbp)
	je	.L23
	cmpl	$9, -12(%rbp)
	je	.L24
	jmp	.L22
.L21:
	movq	$7, -8(%rbp)
	jmp	.L25
.L24:
	movq	$0, -8(%rbp)
	jmp	.L25
.L23:
	movq	$10, -8(%rbp)
	jmp	.L25
.L22:
	movq	$4, -8(%rbp)
	nop
.L25:
	jmp	.L17
.L28:
	nop
.L17:
	jmp	.L26
.L27:
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
