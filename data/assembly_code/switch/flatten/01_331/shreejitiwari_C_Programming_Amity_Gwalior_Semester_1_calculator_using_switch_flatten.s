	.file	"shreejitiwari_C_Programming_Amity_Gwalior_Semester_1_calculator_using_switch_flatten.c"
	.text
	.globl	_TIG_IZ_iYCA_argc
	.bss
	.align 4
	.type	_TIG_IZ_iYCA_argc, @object
	.size	_TIG_IZ_iYCA_argc, 4
_TIG_IZ_iYCA_argc:
	.zero	4
	.globl	_TIG_IZ_iYCA_argv
	.align 8
	.type	_TIG_IZ_iYCA_argv, @object
	.size	_TIG_IZ_iYCA_argv, 8
_TIG_IZ_iYCA_argv:
	.zero	8
	.globl	_TIG_IZ_iYCA_envp
	.align 8
	.type	_TIG_IZ_iYCA_envp, @object
	.size	_TIG_IZ_iYCA_envp, 8
_TIG_IZ_iYCA_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Infinite ..."
.LC2:
	.string	"Invalid input !!"
.LC3:
	.string	"Output for %c  = %.2f "
	.align 8
.LC4:
	.string	"Which operation you want to perform ?\n1) + for Addition\n2) - for Subtraction\n3) * for Product\n4) / for Division\n\nYour choice : "
.LC5:
	.string	"%c"
.LC6:
	.string	"\nEnter a : "
.LC7:
	.string	"%f"
.LC8:
	.string	"Enter b : "
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_iYCA_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_iYCA_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_iYCA_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 89 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-iYCA--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_iYCA_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_iYCA_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_iYCA_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L36:
	cmpq	$18, -16(%rbp)
	ja	.L41
	movq	-16(%rbp), %rax
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
	.long	.L21-.L8
	.long	.L41-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L41-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L41-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L41-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L41-.L8
	.long	.L7-.L8
	.text
.L7:
	movzbl	-33(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L22
	cmpl	$47, %eax
	jg	.L23
	cmpl	$45, %eax
	je	.L24
	cmpl	$45, %eax
	jg	.L23
	cmpl	$42, %eax
	je	.L25
	cmpl	$43, %eax
	je	.L26
	jmp	.L23
.L22:
	movq	$2, -16(%rbp)
	jmp	.L27
.L25:
	movq	$8, -16(%rbp)
	jmp	.L27
.L24:
	movq	$0, -16(%rbp)
	jmp	.L27
.L26:
	movq	$15, -16(%rbp)
	jmp	.L27
.L23:
	movq	$13, -16(%rbp)
	nop
.L27:
	jmp	.L28
.L18:
	movq	$10, -16(%rbp)
	jmp	.L28
.L10:
	movss	-32(%rbp), %xmm1
	movss	-28(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L28
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L37
	jmp	.L40
.L15:
	movss	-32(%rbp), %xmm1
	movss	-28(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L28
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L9:
	pxor	%xmm0, %xmm0
	ucomiss	-20(%rbp), %xmm0
	jp	.L38
	pxor	%xmm0, %xmm0
	ucomiss	-20(%rbp), %xmm0
	je	.L30
.L38:
	movq	$5, -16(%rbp)
	jmp	.L28
.L30:
	movq	$3, -16(%rbp)
	jmp	.L28
.L14:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L28
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -16(%rbp)
	jmp	.L28
.L17:
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rdx
	movzbl	-33(%rbp), %eax
	movsbl	%al, %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-33(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L28
.L21:
	movss	-32(%rbp), %xmm0
	movss	-28(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L28
.L16:
	movss	-32(%rbp), %xmm0
	movss	-28(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L28
.L20:
	movss	-28(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L39
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	je	.L33
.L39:
	movq	$7, -16(%rbp)
	jmp	.L28
.L33:
	movq	$9, -16(%rbp)
	jmp	.L28
.L41:
	nop
.L28:
	jmp	.L36
.L40:
	call	__stack_chk_fail@PLT
.L37:
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
