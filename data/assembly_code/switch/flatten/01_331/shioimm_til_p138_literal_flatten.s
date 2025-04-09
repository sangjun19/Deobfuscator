	.file	"shioimm_til_p138_literal_flatten.c"
	.text
	.local	bar
	.comm	bar,4,1
	.local	foo
	.comm	foo,4,1
	.globl	_TIG_IZ_jJDv_envp
	.bss
	.align 8
	.type	_TIG_IZ_jJDv_envp, @object
	.size	_TIG_IZ_jJDv_envp, 8
_TIG_IZ_jJDv_envp:
	.zero	8
	.globl	_TIG_IZ_jJDv_argc
	.align 4
	.type	_TIG_IZ_jJDv_argc, @object
	.size	_TIG_IZ_jJDv_argc, 4
_TIG_IZ_jJDv_argc:
	.zero	4
	.globl	_TIG_IZ_jJDv_argv
	.align 8
	.type	_TIG_IZ_jJDv_argv, @object
	.size	_TIG_IZ_jJDv_argv, 8
_TIG_IZ_jJDv_argv:
	.zero	8
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
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
	movb	$66, bar(%rip)
	movb	$65, 1+bar(%rip)
	movb	$82, 2+bar(%rip)
	movb	$0, 3+bar(%rip)
	nop
.L2:
	movb	$70, foo(%rip)
	movb	$79, 1+foo(%rip)
	movb	$79, 2+foo(%rip)
	movb	$0, 3+foo(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_jJDv_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_IZ_jJDv_argv(%rip)
	nop
.L5:
	movl	$0, _TIG_IZ_jJDv_argc(%rip)
	nop
	nop
.L6:
.L7:
#APP
# 142 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jJDv--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_jJDv_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_jJDv_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_jJDv_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L13:
	cmpq	$2, -16(%rbp)
	je	.L8
	cmpq	$2, -16(%rbp)
	ja	.L15
	cmpq	$0, -16(%rbp)
	je	.L10
	cmpq	$1, -16(%rbp)
	jne	.L15
	movq	$0, -16(%rbp)
	jmp	.L11
.L10:
	movl	$100, %edi
	call	lit
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L11
.L8:
	movl	$0, %eax
	jmp	.L14
.L15:
	nop
.L11:
	jmp	.L13
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	lit
	.type	lit, @function
lit:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L29:
	cmpq	$4, -8(%rbp)
	ja	.L30
	movq	-8(%rbp), %rax
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
	.long	.L23-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L18:
	movl	$0, %eax
	jmp	.L24
.L22:
	cmpl	$100, -20(%rbp)
	je	.L25
	cmpl	$200, -20(%rbp)
	jne	.L26
	movq	$3, -8(%rbp)
	jmp	.L27
.L25:
	movq	$0, -8(%rbp)
	jmp	.L27
.L26:
	movq	$2, -8(%rbp)
	nop
.L27:
	jmp	.L28
.L20:
	leaq	bar(%rip), %rax
	jmp	.L24
.L23:
	leaq	foo(%rip), %rax
	jmp	.L24
.L21:
	movq	$4, -8(%rbp)
	jmp	.L28
.L30:
	nop
.L28:
	jmp	.L29
.L24:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	lit, .-lit
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
