	.file	"Nil369_Akash_C_Code_NOTES_01_Basic_calculator_flatten.c"
	.text
	.globl	_TIG_IZ_CJH7_envp
	.bss
	.align 8
	.type	_TIG_IZ_CJH7_envp, @object
	.size	_TIG_IZ_CJH7_envp, 8
_TIG_IZ_CJH7_envp:
	.zero	8
	.globl	_TIG_IZ_CJH7_argc
	.align 4
	.type	_TIG_IZ_CJH7_argc, @object
	.size	_TIG_IZ_CJH7_argc, 4
_TIG_IZ_CJH7_argc:
	.zero	4
	.globl	_TIG_IZ_CJH7_argv
	.align 8
	.type	_TIG_IZ_CJH7_argv, @object
	.size	_TIG_IZ_CJH7_argv, 8
_TIG_IZ_CJH7_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The division is: %f"
.LC1:
	.string	"\nEnter 1st number: "
.LC2:
	.string	"%f"
.LC3:
	.string	"\nOperation to perform: "
.LC4:
	.string	" %c"
.LC5:
	.string	"\nEnter 2nd number: "
.LC6:
	.string	"The sum is: %f"
	.align 8
.LC7:
	.string	"You haven't used a right operator.Use among +,-,*,/"
.LC8:
	.string	"The diff is: %f"
.LC9:
	.string	"The product is: %f"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_CJH7_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_CJH7_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_CJH7_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CJH7--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_CJH7_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_CJH7_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_CJH7_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L30:
	cmpq	$22, -16(%rbp)
	ja	.L33
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
	.long	.L33-.L8
	.long	.L19-.L8
	.long	.L33-.L8
	.long	.L18-.L8
	.long	.L33-.L8
	.long	.L17-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L16-.L8
	.long	.L33-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movss	-32(%rbp), %xmm0
	movss	-28(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L20
.L14:
	movl	$0, -24(%rbp)
	movl	$0, -24(%rbp)
	movq	$19, -16(%rbp)
	jmp	.L20
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-33(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L20
.L18:
	movss	-32(%rbp), %xmm1
	movss	-28(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm3, %xmm3
	cvtss2sd	-20(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L20
.L12:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L20
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L32
.L15:
	addl	$1, -24(%rbp)
	movq	$19, -16(%rbp)
	jmp	.L20
.L16:
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
	movq	$15, -16(%rbp)
	jmp	.L27
.L25:
	movq	$20, -16(%rbp)
	jmp	.L27
.L24:
	movq	$22, -16(%rbp)
	jmp	.L27
.L26:
	movq	$3, -16(%rbp)
	jmp	.L27
.L23:
	movq	$16, -16(%rbp)
	nop
.L27:
	jmp	.L20
.L11:
	cmpl	$9, -24(%rbp)
	jg	.L28
	movq	$1, -16(%rbp)
	jmp	.L20
.L28:
	movq	$21, -16(%rbp)
	jmp	.L20
.L7:
	movss	-32(%rbp), %xmm0
	movss	-28(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm4, %xmm4
	cvtss2sd	-20(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L20
.L17:
	movq	$12, -16(%rbp)
	jmp	.L20
.L10:
	movss	-32(%rbp), %xmm1
	movss	-28(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm5, %xmm5
	cvtss2sd	-20(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L20
.L33:
	nop
.L20:
	jmp	.L30
.L32:
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
