	.file	"savanthreddymarthala_c_language_f_flatten.c"
	.text
	.globl	_TIG_IZ_jWQw_argv
	.bss
	.align 8
	.type	_TIG_IZ_jWQw_argv, @object
	.size	_TIG_IZ_jWQw_argv, 8
_TIG_IZ_jWQw_argv:
	.zero	8
	.globl	_TIG_IZ_jWQw_argc
	.align 4
	.type	_TIG_IZ_jWQw_argc, @object
	.size	_TIG_IZ_jWQw_argc, 4
_TIG_IZ_jWQw_argc:
	.zero	4
	.globl	_TIG_IZ_jWQw_envp
	.align 8
	.type	_TIG_IZ_jWQw_envp, @object
	.size	_TIG_IZ_jWQw_envp, 8
_TIG_IZ_jWQw_envp:
	.zero	8
	.text
	.globl	fib
	.type	fib, @function
fib:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L16:
	cmpq	$6, -8(%rbp)
	ja	.L17
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
	.long	.L17-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	movl	$1, %eax
	jmp	.L10
.L7:
	cmpl	$0, -20(%rbp)
	jne	.L11
	movq	$0, -8(%rbp)
	jmp	.L13
.L11:
	movq	$2, -8(%rbp)
	jmp	.L13
.L3:
	movl	-20(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	subl	$2, %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L13
.L5:
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	jmp	.L10
.L9:
	movl	$0, %eax
	jmp	.L10
.L8:
	cmpl	$1, -20(%rbp)
	jne	.L14
	movq	$4, -8(%rbp)
	jmp	.L13
.L14:
	movq	$6, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L16
.L10:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	fib, .-fib
	.section	.rodata
.LC0:
	.string	"enter the value off n"
.LC1:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_jWQw_envp(%rip)
	nop
.L19:
	movq	$0, _TIG_IZ_jWQw_argv(%rip)
	nop
.L20:
	movl	$0, _TIG_IZ_jWQw_argc(%rip)
	nop
	nop
.L21:
.L22:
#APP
# 88 "savanthreddymarthala_c_language_f.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jWQw--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_jWQw_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_jWQw_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_jWQw_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L34:
	cmpq	$7, -16(%rbp)
	ja	.L37
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L37-.L25
	.long	.L26-.L25
	.long	.L37-.L25
	.long	.L37-.L25
	.long	.L24-.L25
	.text
.L26:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L30
.L28:
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L30
.L29:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L35
	jmp	.L36
.L24:
	movq	$4, -16(%rbp)
	jmp	.L30
.L27:
	movl	-28(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jge	.L32
	movq	$1, -16(%rbp)
	jmp	.L30
.L32:
	movq	$0, -16(%rbp)
	jmp	.L30
.L37:
	nop
.L30:
	jmp	.L34
.L36:
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
