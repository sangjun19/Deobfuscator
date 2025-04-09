	.file	"boarrr_C-Programming_FunctionReturn_flatten.c"
	.text
	.globl	_TIG_IZ_OE81_envp
	.bss
	.align 8
	.type	_TIG_IZ_OE81_envp, @object
	.size	_TIG_IZ_OE81_envp, 8
_TIG_IZ_OE81_envp:
	.zero	8
	.globl	_TIG_IZ_OE81_argv
	.align 8
	.type	_TIG_IZ_OE81_argv, @object
	.size	_TIG_IZ_OE81_argv, 8
_TIG_IZ_OE81_argv:
	.zero	8
	.globl	_TIG_IZ_OE81_argc
	.align 4
	.type	_TIG_IZ_OE81_argc, @object
	.size	_TIG_IZ_OE81_argc, 4
_TIG_IZ_OE81_argc:
	.zero	4
	.text
	.globl	find_minimum
	.type	find_minimum, @function
find_minimum:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L10
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L10
	movl	-24(%rbp), %eax
	jmp	.L5
.L4:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L6
	movq	$2, -8(%rbp)
	jmp	.L8
.L6:
	movq	$1, -8(%rbp)
	jmp	.L8
.L2:
	movl	-20(%rbp), %eax
	jmp	.L5
.L10:
	nop
.L8:
	jmp	.L9
.L5:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	find_minimum, .-find_minimum
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter any two different whole numbers"
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"\nThe minimum of %d and %d is %d"
	.align 8
.LC3:
	.string	"\nNumbers are equal. Try again\n"
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
	movq	$0, _TIG_IZ_OE81_envp(%rip)
	nop
.L12:
	movq	$0, _TIG_IZ_OE81_argv(%rip)
	nop
.L13:
	movl	$0, _TIG_IZ_OE81_argc(%rip)
	nop
	nop
.L14:
.L15:
#APP
# 187 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-OE81--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_OE81_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_OE81_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_OE81_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L31:
	cmpq	$9, -16(%rbp)
	ja	.L34
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L24-.L18
	.long	.L23-.L18
	.long	.L22-.L18
	.long	.L21-.L18
	.long	.L34-.L18
	.long	.L34-.L18
	.long	.L20-.L18
	.long	.L34-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L25
.L23:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L26
	movq	$8, -16(%rbp)
	jmp	.L25
.L26:
	movq	$6, -16(%rbp)
	jmp	.L25
.L21:
	movl	$0, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L25
.L17:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L28
	movq	$2, -16(%rbp)
	jmp	.L25
.L28:
	movq	$1, -16(%rbp)
	jmp	.L25
.L20:
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	find_minimum
	movl	%eax, -20(%rbp)
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	-20(%rbp), %ecx
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L25
.L24:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L32
	jmp	.L33
.L22:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L34:
	nop
.L25:
	jmp	.L31
.L33:
	call	__stack_chk_fail@PLT
.L32:
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
