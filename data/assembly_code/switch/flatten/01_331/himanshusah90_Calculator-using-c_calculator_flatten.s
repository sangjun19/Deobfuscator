	.file	"himanshusah90_Calculator-using-c_calculator_flatten.c"
	.text
	.globl	_TIG_IZ_4Poj_argc
	.bss
	.align 4
	.type	_TIG_IZ_4Poj_argc, @object
	.size	_TIG_IZ_4Poj_argc, 4
_TIG_IZ_4Poj_argc:
	.zero	4
	.globl	_TIG_IZ_4Poj_argv
	.align 8
	.type	_TIG_IZ_4Poj_argv, @object
	.size	_TIG_IZ_4Poj_argv, 8
_TIG_IZ_4Poj_argv:
	.zero	8
	.globl	_TIG_IZ_4Poj_envp
	.align 8
	.type	_TIG_IZ_4Poj_envp, @object
	.size	_TIG_IZ_4Poj_envp, 8
_TIG_IZ_4Poj_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Error! Operator is not correct."
.LC3:
	.string	"Error! Division by zero."
	.text
	.globl	simpleCalc
	.type	simpleCalc, @function
simpleCalc:
.LFB5:
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
	movq	$1, -16(%rbp)
.L26:
	cmpq	$14, -16(%rbp)
	ja	.L28
	movq	-16(%rbp), %rax
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
	.long	.L28-.L4
	.long	.L14-.L4
	.long	.L28-.L4
	.long	.L13-.L4
	.long	.L28-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L28-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L6:
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	jp	.L27
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	je	.L16
.L27:
	movq	$13, -16(%rbp)
	jmp	.L15
.L16:
	movq	$5, -16(%rbp)
	jmp	.L15
.L9:
	movsd	-24(%rbp), %xmm0
	subsd	-32(%rbp), %xmm0
	movsd	%xmm0, -8(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L15
.L14:
	movsbl	-36(%rbp), %eax
	cmpl	$47, %eax
	je	.L19
	cmpl	$47, %eax
	jg	.L20
	cmpl	$45, %eax
	je	.L21
	cmpl	$45, %eax
	jg	.L20
	cmpl	$42, %eax
	je	.L22
	cmpl	$43, %eax
	je	.L23
	jmp	.L20
.L19:
	movq	$12, -16(%rbp)
	jmp	.L24
.L22:
	movq	$3, -16(%rbp)
	jmp	.L24
.L21:
	movq	$8, -16(%rbp)
	jmp	.L24
.L23:
	movq	$7, -16(%rbp)
	jmp	.L24
.L20:
	movq	$14, -16(%rbp)
	nop
.L24:
	jmp	.L15
.L13:
	movsd	-24(%rbp), %xmm0
	mulsd	-32(%rbp), %xmm0
	movsd	%xmm0, -8(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L15
.L8:
	movsd	.LC2(%rip), %xmm0
	jmp	.L25
.L5:
	movsd	-24(%rbp), %xmm0
	divsd	-32(%rbp), %xmm0
	movsd	%xmm0, -8(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L15
.L11:
	pxor	%xmm0, %xmm0
	jmp	.L25
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -16(%rbp)
	jmp	.L15
.L7:
	movsd	.LC2(%rip), %xmm0
	jmp	.L25
.L10:
	movsd	-24(%rbp), %xmm0
	addsd	-32(%rbp), %xmm0
	movsd	%xmm0, -8(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L15
.L28:
	nop
.L15:
	jmp	.L26
.L25:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	simpleCalc, .-simpleCalc
	.section	.rodata
	.align 8
.LC4:
	.string	"Enter an operator (+, -, *, /): "
.LC5:
	.string	"%c"
.LC6:
	.string	"Enter two operands: "
.LC7:
	.string	"%lf %lf"
.LC8:
	.string	"Result: %.2lf\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_4Poj_envp(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_4Poj_argv(%rip)
	nop
.L31:
	movl	$0, _TIG_IZ_4Poj_argc(%rip)
	nop
	nop
.L32:
.L33:
#APP
# 69 "himanshusah90_Calculator-using-c_calculator.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4Poj--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_4Poj_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_4Poj_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_4Poj_envp(%rip)
	nop
	movq	$2, -24(%rbp)
.L39:
	cmpq	$2, -24(%rbp)
	je	.L34
	cmpq	$2, -24(%rbp)
	ja	.L42
	cmpq	$0, -24(%rbp)
	je	.L36
	cmpq	$1, -24(%rbp)
	jne	.L42
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	jmp	.L41
.L36:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-41(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movzbl	-41(%rbp), %eax
	movsbl	%al, %edx
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movl	%edx, %edi
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	simpleCalc
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -24(%rbp)
	jmp	.L38
.L34:
	movq	$0, -24(%rbp)
	jmp	.L38
.L42:
	nop
.L38:
	jmp	.L39
.L41:
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC2:
	.long	0
	.long	-1074790400
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
