	.file	"kernel-bz_linux-kernel-test_switch-case_flatten.c"
	.text
	.globl	_TIG_IZ_QO4h_argc
	.bss
	.align 4
	.type	_TIG_IZ_QO4h_argc, @object
	.size	_TIG_IZ_QO4h_argc, 4
_TIG_IZ_QO4h_argc:
	.zero	4
	.globl	_TIG_IZ_QO4h_argv
	.align 8
	.type	_TIG_IZ_QO4h_argv, @object
	.size	_TIG_IZ_QO4h_argv, 8
_TIG_IZ_QO4h_argv:
	.zero	8
	.globl	_TIG_IZ_QO4h_envp
	.align 8
	.type	_TIG_IZ_QO4h_envp, @object
	.size	_TIG_IZ_QO4h_envp, 8
_TIG_IZ_QO4h_envp:
	.zero	8
	.text
	.type	scase2, @function
scase2:
.LFB3:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L17:
	cmpq	$5, -8(%rbp)
	ja	.L19
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movq	$2, -8(%rbp)
	jmp	.L10
.L8:
	cmpl	$3, -20(%rbp)
	je	.L11
	cmpl	$3, -20(%rbp)
	jg	.L12
	cmpl	$0, -20(%rbp)
	je	.L13
	cmpl	$0, -20(%rbp)
	js	.L12
	movl	-20(%rbp), %eax
	subl	$1, %eax
	cmpl	$1, %eax
	ja	.L12
	jmp	.L18
.L11:
	movq	$3, -8(%rbp)
	jmp	.L15
.L18:
	movq	$5, -8(%rbp)
	jmp	.L15
.L13:
	movq	$0, -8(%rbp)
	jmp	.L15
.L12:
	movq	$4, -8(%rbp)
	nop
.L15:
	jmp	.L10
.L6:
	movl	$3, %eax
	jmp	.L16
.L3:
	movl	$2, %eax
	jmp	.L16
.L9:
	movl	$0, %eax
	jmp	.L16
.L7:
	movl	$0, %eax
	jmp	.L16
.L19:
	nop
.L10:
	jmp	.L17
.L16:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	scase2, .-scase2
	.type	scase1, @function
scase1:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$2, -8(%rbp)
.L31:
	cmpq	$4, -8(%rbp)
	je	.L21
	cmpq	$4, -8(%rbp)
	ja	.L32
	cmpq	$3, -8(%rbp)
	je	.L23
	cmpq	$3, -8(%rbp)
	ja	.L32
	cmpq	$1, -8(%rbp)
	je	.L24
	cmpq	$2, -8(%rbp)
	je	.L25
	jmp	.L32
.L21:
	movl	$0, %eax
	jmp	.L26
.L24:
	movl	$3, %eax
	jmp	.L26
.L23:
	movl	$1, %eax
	jmp	.L26
.L25:
	cmpl	$0, -20(%rbp)
	je	.L27
	cmpl	$1, -20(%rbp)
	jne	.L28
	movq	$3, -8(%rbp)
	jmp	.L29
.L27:
	movq	$4, -8(%rbp)
	jmp	.L29
.L28:
	movq	$1, -8(%rbp)
	nop
.L29:
	jmp	.L30
.L32:
	nop
.L30:
	jmp	.L31
.L26:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	scase1, .-scase1
	.section	.rodata
.LC0:
	.string	"ret=%d\n"
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
	movq	$0, _TIG_IZ_QO4h_envp(%rip)
	nop
.L34:
	movq	$0, _TIG_IZ_QO4h_argv(%rip)
	nop
.L35:
	movl	$0, _TIG_IZ_QO4h_argc(%rip)
	nop
	nop
.L36:
.L37:
#APP
# 254 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-QO4h--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_QO4h_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_QO4h_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_QO4h_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L43:
	cmpq	$2, -8(%rbp)
	je	.L38
	cmpq	$2, -8(%rbp)
	ja	.L45
	cmpq	$0, -8(%rbp)
	je	.L40
	cmpq	$1, -8(%rbp)
	jne	.L45
	movl	$1, %edi
	call	scase1
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$2, %edi
	call	scase2
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L41
.L40:
	movl	$0, %eax
	jmp	.L44
.L38:
	movq	$1, -8(%rbp)
	jmp	.L41
.L45:
	nop
.L41:
	jmp	.L43
.L44:
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
