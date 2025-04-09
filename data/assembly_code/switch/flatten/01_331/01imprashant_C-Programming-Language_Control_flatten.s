	.file	"01imprashant_C-Programming-Language_Control_flatten.c"
	.text
	.globl	_TIG_IZ_eXkE_envp
	.bss
	.align 8
	.type	_TIG_IZ_eXkE_envp, @object
	.size	_TIG_IZ_eXkE_envp, 8
_TIG_IZ_eXkE_envp:
	.zero	8
	.globl	_TIG_IZ_eXkE_argv
	.align 8
	.type	_TIG_IZ_eXkE_argv, @object
	.size	_TIG_IZ_eXkE_argv, 8
_TIG_IZ_eXkE_argv:
	.zero	8
	.globl	_TIG_IZ_eXkE_argc
	.align 4
	.type	_TIG_IZ_eXkE_argc, @object
	.size	_TIG_IZ_eXkE_argc, 4
_TIG_IZ_eXkE_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"default"
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
	movq	$0, _TIG_IZ_eXkE_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_eXkE_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_eXkE_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-eXkE--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_eXkE_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_eXkE_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_eXkE_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L17:
	cmpq	$6, -8(%rbp)
	ja	.L18
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L19-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L15
.L13:
	movb	$65, -9(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L15
.L7:
	movl	$50, %edi
	call	putchar@PLT
	movq	$0, -8(%rbp)
	jmp	.L15
.L9:
	movq	$4, -8(%rbp)
	nop
	jmp	.L15
.L14:
	movl	$51, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L15
.L12:
	movl	$49, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L15
.L18:
	nop
.L15:
	jmp	.L17
.L19:
	nop
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
