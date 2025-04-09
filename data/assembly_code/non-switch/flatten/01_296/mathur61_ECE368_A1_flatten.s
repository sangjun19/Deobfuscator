	.file	"mathur61_ECE368_A1_flatten.c"
	.text
	.globl	_TIG_IZ_8Xc8_argc
	.bss
	.align 4
	.type	_TIG_IZ_8Xc8_argc, @object
	.size	_TIG_IZ_8Xc8_argc, 4
_TIG_IZ_8Xc8_argc:
	.zero	4
	.globl	_TIG_IZ_8Xc8_envp
	.align 8
	.type	_TIG_IZ_8Xc8_envp, @object
	.size	_TIG_IZ_8Xc8_envp, 8
_TIG_IZ_8Xc8_envp:
	.zero	8
	.globl	_TIG_IZ_8Xc8_argv
	.align 8
	.type	_TIG_IZ_8Xc8_argv, @object
	.size	_TIG_IZ_8Xc8_argv, 8
_TIG_IZ_8Xc8_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"%d quarter(s), %d dime(s), %d nickel(s), %d pennie(s)\n"
	.text
	.globl	print_combinations
	.type	print_combinations, @function
print_combinations:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$6, -8(%rbp)
.L22:
	cmpq	$18, -8(%rbp)
	ja	.L23
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
	.long	.L13-.L4
	.long	.L23-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L23-.L4
	.long	.L23-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L24-.L4
	.long	.L7-.L4
	.long	.L23-.L4
	.long	.L23-.L4
	.long	.L23-.L4
	.long	.L23-.L4
	.long	.L6-.L4
	.long	.L23-.L4
	.long	.L5-.L4
	.long	.L23-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-24(%rbp), %eax
	imull	$-10, %eax, %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	%edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L14
.L6:
	movl	-16(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	negl	%eax
	movl	-20(%rbp), %edx
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %esi
	movl	-16(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	%esi, %r8d
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L14
.L11:
	subl	$1, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L14
.L5:
	movl	-32(%rbp), %eax
	imull	$-25, %eax, %edx
	movl	-36(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L14
.L7:
	cmpl	$0, -16(%rbp)
	js	.L16
	movq	$14, -8(%rbp)
	jmp	.L14
.L16:
	movq	$3, -8(%rbp)
	jmp	.L14
.L10:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1374389535, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$3, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -32(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L13:
	cmpl	$0, -32(%rbp)
	js	.L18
	movq	$16, -8(%rbp)
	jmp	.L14
.L18:
	movq	$8, -8(%rbp)
	jmp	.L14
.L9:
	cmpl	$0, -24(%rbp)
	js	.L20
	movq	$18, -8(%rbp)
	jmp	.L14
.L20:
	movq	$2, -8(%rbp)
	jmp	.L14
.L12:
	subl	$1, -32(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L23:
	nop
.L14:
	jmp	.L22
.L24:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	print_combinations, .-print_combinations
	.section	.rodata
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
	movq	$0, _TIG_IZ_8Xc8_envp(%rip)
	nop
.L26:
	movq	$0, _TIG_IZ_8Xc8_argv(%rip)
	nop
.L27:
	movl	$0, _TIG_IZ_8Xc8_argc(%rip)
	nop
	nop
.L28:
.L29:
#APP
# 54 "mathur61_ECE368_A1.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8Xc8--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_8Xc8_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_8Xc8_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_8Xc8_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L35:
	cmpq	$2, -16(%rbp)
	je	.L30
	cmpq	$2, -16(%rbp)
	ja	.L38
	cmpq	$0, -16(%rbp)
	je	.L32
	cmpq	$1, -16(%rbp)
	jne	.L38
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	print_combinations
	movq	$2, -16(%rbp)
	jmp	.L33
.L32:
	movq	$1, -16(%rbp)
	jmp	.L33
.L30:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L36
	jmp	.L37
.L38:
	nop
.L33:
	jmp	.L35
.L37:
	call	__stack_chk_fail@PLT
.L36:
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
