	.file	"KPRenu_python_3_flatten.c"
	.text
	.globl	_TIG_IZ_cwXu_argc
	.bss
	.align 4
	.type	_TIG_IZ_cwXu_argc, @object
	.size	_TIG_IZ_cwXu_argc, 4
_TIG_IZ_cwXu_argc:
	.zero	4
	.globl	_TIG_IZ_cwXu_argv
	.align 8
	.type	_TIG_IZ_cwXu_argv, @object
	.size	_TIG_IZ_cwXu_argv, 8
_TIG_IZ_cwXu_argv:
	.zero	8
	.globl	_TIG_IZ_cwXu_envp
	.align 8
	.type	_TIG_IZ_cwXu_envp, @object
	.size	_TIG_IZ_cwXu_envp, 8
_TIG_IZ_cwXu_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\n The consumed value is %d"
	.align 8
.LC1:
	.string	"\n 1. Produce \t 2. Consume \t3. Exit "
.LC2:
	.string	"\n Enter your choice: "
.LC3:
	.string	"%d"
.LC4:
	.string	"\n Buffer is Empty"
.LC5:
	.string	"\n Enter the value: "
.LC6:
	.string	"\n Buffer is Full"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_cwXu_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_cwXu_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_cwXu_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-cwXu--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_cwXu_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_cwXu_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_cwXu_envp(%rip)
	nop
	movq	$4, -56(%rbp)
.L32:
	cmpq	$19, -56(%rbp)
	ja	.L35
	movq	-56(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L35-.L8
	.long	.L11-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L36-.L8
	.text
.L16:
	movq	$5, -56(%rbp)
	jmp	.L21
.L10:
	movl	-68(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jne	.L22
	movq	$16, -56(%rbp)
	jmp	.L21
.L22:
	movq	$8, -56(%rbp)
	jmp	.L21
.L13:
	movl	-64(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-64(%rbp), %eax
	addl	$1, %eax
	cltd
	idivl	-72(%rbp)
	movl	%edx, -64(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L21
.L19:
	movl	-76(%rbp), %eax
	cmpl	$1, %eax
	je	.L24
	cmpl	$2, %eax
	jne	.L25
	movq	$15, -56(%rbp)
	jmp	.L26
.L24:
	movq	$9, -56(%rbp)
	jmp	.L26
.L25:
	movq	$2, -56(%rbp)
	nop
.L26:
	jmp	.L21
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -56(%rbp)
	jmp	.L21
.L9:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -56(%rbp)
	jmp	.L21
.L11:
	movl	-76(%rbp), %eax
	cmpl	$3, %eax
	je	.L27
	movq	$3, -56(%rbp)
	jmp	.L21
.L27:
	movq	$19, -56(%rbp)
	jmp	.L21
.L12:
	movl	-68(%rbp), %eax
	addl	$1, %eax
	cltd
	idivl	-72(%rbp)
	movl	%edx, %eax
	cmpl	%eax, -64(%rbp)
	jne	.L29
	movq	$7, -56(%rbp)
	jmp	.L21
.L29:
	movq	$0, -56(%rbp)
	jmp	.L21
.L15:
	movl	$0, -76(%rbp)
	movl	$0, -68(%rbp)
	movl	$0, -64(%rbp)
	movl	$10, -72(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L21
.L20:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-80(%rbp), %edx
	movl	-68(%rbp), %eax
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	movl	-68(%rbp), %eax
	addl	$1, %eax
	cltd
	idivl	-72(%rbp)
	movl	%edx, -68(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L21
.L14:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -56(%rbp)
	jmp	.L21
.L18:
	movq	$11, -56(%rbp)
	jmp	.L21
.L35:
	nop
.L21:
	jmp	.L32
.L36:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L34
	call	__stack_chk_fail@PLT
.L34:
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
