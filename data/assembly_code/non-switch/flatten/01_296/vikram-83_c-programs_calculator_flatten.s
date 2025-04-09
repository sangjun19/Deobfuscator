	.file	"vikram-83_c-programs_calculator_flatten.c"
	.text
	.globl	_TIG_IZ_DEsq_envp
	.bss
	.align 8
	.type	_TIG_IZ_DEsq_envp, @object
	.size	_TIG_IZ_DEsq_envp, 8
_TIG_IZ_DEsq_envp:
	.zero	8
	.globl	_TIG_IZ_DEsq_argv
	.align 8
	.type	_TIG_IZ_DEsq_argv, @object
	.size	_TIG_IZ_DEsq_argv, 8
_TIG_IZ_DEsq_argv:
	.zero	8
	.globl	_TIG_IZ_DEsq_argc
	.align 4
	.type	_TIG_IZ_DEsq_argc, @object
	.size	_TIG_IZ_DEsq_argc, 4
_TIG_IZ_DEsq_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter an operator (+, -, *, /): "
.LC1:
	.string	"%c"
.LC2:
	.string	"Enter two operands: "
.LC3:
	.string	"%lf %lf"
.LC4:
	.string	"Result: %.2lf\n"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_DEsq_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_DEsq_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_DEsq_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-DEsq--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_DEsq_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_DEsq_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_DEsq_envp(%rip)
	nop
	movq	$1, -32(%rbp)
.L11:
	cmpq	$2, -32(%rbp)
	je	.L6
	cmpq	$2, -32(%rbp)
	ja	.L14
	cmpq	$0, -32(%rbp)
	je	.L8
	cmpq	$1, -32(%rbp)
	jne	.L14
	movq	$2, -32(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-49(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movzbl	-49(%rbp), %eax
	movsbl	%al, %edx
	movsd	-40(%rbp), %xmm0
	movq	-48(%rbp), %rax
	movl	%edx, %edi
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	simpleCalc
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movsd	%xmm0, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$0, -32(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC6:
	.string	"Error! Division by zero."
	.align 8
.LC8:
	.string	"Error! Operator is not correct."
	.text
	.globl	simpleCalc
	.type	simpleCalc, @function
simpleCalc:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movsd	%xmm0, -24(%rbp)
	movsd	%xmm1, -32(%rbp)
	movl	%edi, %eax
	movb	%al, -36(%rbp)
	movq	$10, -8(%rbp)
.L45:
	cmpq	$13, -8(%rbp)
	ja	.L47
	movq	-8(%rbp), %rax
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
	.long	.L31-.L18
	.long	.L30-.L18
	.long	.L29-.L18
	.long	.L28-.L18
	.long	.L27-.L18
	.long	.L26-.L18
	.long	.L25-.L18
	.long	.L24-.L18
	.long	.L23-.L18
	.long	.L22-.L18
	.long	.L21-.L18
	.long	.L20-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L27:
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	jp	.L46
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	je	.L32
.L46:
	movq	$11, -8(%rbp)
	jmp	.L35
.L32:
	movq	$12, -8(%rbp)
	jmp	.L35
.L19:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -8(%rbp)
	jmp	.L35
.L23:
	movsd	-24(%rbp), %xmm0
	mulsd	-32(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L35
.L30:
	cmpb	$45, -36(%rbp)
	jne	.L36
	movq	$7, -8(%rbp)
	jmp	.L35
.L36:
	movq	$5, -8(%rbp)
	jmp	.L35
.L28:
	movsd	-24(%rbp), %xmm0
	addsd	-32(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L35
.L20:
	movsd	-24(%rbp), %xmm0
	divsd	-32(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L35
.L22:
	movsd	.LC7(%rip), %xmm0
	jmp	.L38
.L17:
	movsd	.LC7(%rip), %xmm0
	jmp	.L38
.L25:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L35
.L26:
	cmpb	$42, -36(%rbp)
	jne	.L39
	movq	$8, -8(%rbp)
	jmp	.L35
.L39:
	movq	$0, -8(%rbp)
	jmp	.L35
.L21:
	cmpb	$43, -36(%rbp)
	jne	.L41
	movq	$3, -8(%rbp)
	jmp	.L35
.L41:
	movq	$1, -8(%rbp)
	jmp	.L35
.L31:
	cmpb	$47, -36(%rbp)
	jne	.L43
	movq	$4, -8(%rbp)
	jmp	.L35
.L43:
	movq	$6, -8(%rbp)
	jmp	.L35
.L24:
	movsd	-24(%rbp), %xmm0
	subsd	-32(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L35
.L29:
	pxor	%xmm0, %xmm0
	cvtsi2sdl	-12(%rbp), %xmm0
	jmp	.L38
.L47:
	nop
.L35:
	jmp	.L45
.L38:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	simpleCalc, .-simpleCalc
	.section	.rodata
	.align 8
.LC7:
	.long	0
	.long	-1042284544
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
