	.file	"gaurav-rijal_C_Programming_Codes_3.2..ForLoop2_flatten.c"
	.text
	.globl	_TIG_IZ_4GVT_argv
	.bss
	.align 8
	.type	_TIG_IZ_4GVT_argv, @object
	.size	_TIG_IZ_4GVT_argv, 8
_TIG_IZ_4GVT_argv:
	.zero	8
	.globl	_TIG_IZ_4GVT_envp
	.align 8
	.type	_TIG_IZ_4GVT_envp, @object
	.size	_TIG_IZ_4GVT_envp, 8
_TIG_IZ_4GVT_envp:
	.zero	8
	.globl	_TIG_IZ_4GVT_argc
	.align 4
	.type	_TIG_IZ_4GVT_argc, @object
	.size	_TIG_IZ_4GVT_argc, 4
_TIG_IZ_4GVT_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d \t"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movq	$0, _TIG_IZ_4GVT_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_4GVT_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_4GVT_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4GVT--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_4GVT_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_4GVT_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_4GVT_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L20:
	cmpq	$7, -8(%rbp)
	ja	.L22
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L14
.L12:
	movl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L14
.L10:
	movl	$0, %eax
	jmp	.L21
.L13:
	movl	-12(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L16
	movq	$4, -8(%rbp)
	jmp	.L14
.L16:
	movq	$7, -8(%rbp)
	jmp	.L14
.L7:
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L14
.L11:
	cmpl	$100, -12(%rbp)
	jg	.L18
	movq	$0, -8(%rbp)
	jmp	.L14
.L18:
	movq	$3, -8(%rbp)
	jmp	.L14
.L22:
	nop
.L14:
	jmp	.L20
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
