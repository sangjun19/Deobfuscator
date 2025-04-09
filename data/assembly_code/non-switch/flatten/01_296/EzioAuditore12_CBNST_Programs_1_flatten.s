	.file	"EzioAuditore12_CBNST_Programs_1_flatten.c"
	.text
	.globl	_TIG_IZ_hXqZ_envp
	.bss
	.align 8
	.type	_TIG_IZ_hXqZ_envp, @object
	.size	_TIG_IZ_hXqZ_envp, 8
_TIG_IZ_hXqZ_envp:
	.zero	8
	.globl	_TIG_IZ_hXqZ_argc
	.align 4
	.type	_TIG_IZ_hXqZ_argc, @object
	.size	_TIG_IZ_hXqZ_argc, 4
_TIG_IZ_hXqZ_argc:
	.zero	4
	.globl	_TIG_IZ_hXqZ_argv
	.align 8
	.type	_TIG_IZ_hXqZ_argv, @object
	.size	_TIG_IZ_hXqZ_argv, 8
_TIG_IZ_hXqZ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Root=%f  Total Iterations=%d"
.LC1:
	.string	"Roots Lie between %f and %f\n"
.LC3:
	.string	"Roots are Invalid"
.LC4:
	.string	"Iterations=%d  Roots=%f\n"
	.align 8
.LC5:
	.string	"Enter Maximum no of Iterations"
.LC6:
	.string	"%d"
	.align 8
.LC7:
	.string	"Enter the value of x1 and x2(starting boundary)"
.LC8:
	.string	"%f%f"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
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
	movq	$0, _TIG_IZ_hXqZ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_hXqZ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_hXqZ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 115 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-hXqZ--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_hXqZ_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_hXqZ_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_hXqZ_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L38:
	cmpq	$25, -16(%rbp)
	ja	.L47
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
	.long	.L47-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L47-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L47-.L8
	.long	.L18-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L47-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L47-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L39
	jmp	.L43
.L7:
	subl	$1, -48(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-44(%rbp), %xmm2
	movq	%xmm2, %rax
	movl	-48(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$18, -16(%rbp)
	jmp	.L26
.L21:
	movss	-52(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-56(%rbp), %xmm1
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm1, %xmm3
	movq	%xmm3, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L26
.L17:
	movss	-40(%rbp), %xmm0
	mulss	-36(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L44
	movq	$15, -16(%rbp)
	jmp	.L26
.L44:
	movq	$4, -16(%rbp)
	jmp	.L26
.L16:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$21, -16(%rbp)
	jmp	.L26
.L24:
	movl	-60(%rbp), %eax
	cmpl	%eax, -48(%rbp)
	jg	.L30
	movq	$6, -16(%rbp)
	jmp	.L26
.L30:
	movq	$25, -16(%rbp)
	jmp	.L26
.L10:
	pxor	%xmm4, %xmm4
	cvtss2sd	-44(%rbp), %xmm4
	movq	%xmm4, %rdx
	movl	-48(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -48(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L26
.L22:
	movq	$24, -16(%rbp)
	jmp	.L26
.L9:
	movl	$1, -48(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-60(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L11:
	movq	$7, -16(%rbp)
	jmp	.L26
.L18:
	movss	-32(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-28(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L45
	movq	$20, -16(%rbp)
	jmp	.L26
.L45:
	movq	$23, -16(%rbp)
	jmp	.L26
.L13:
	movss	-44(%rbp), %xmm0
	movss	%xmm0, -52(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L26
.L15:
	movl	-44(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -32(%rbp)
	movl	-52(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L20:
	movss	-52(%rbp), %xmm0
	movl	-56(%rbp), %eax
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	bisect
	movd	%xmm0, %eax
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -24(%rbp)
	movl	-56(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L26
.L19:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rdx
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-56(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -40(%rbp)
	movl	-52(%rbp), %eax
	movd	%eax, %xmm0
	call	findValueAt
	movd	%xmm0, %eax
	movl	%eax, -36(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L26
.L23:
	movss	-24(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-20(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L46
	movq	$19, -16(%rbp)
	jmp	.L26
.L46:
	movq	$17, -16(%rbp)
	jmp	.L26
.L12:
	movss	-44(%rbp), %xmm0
	movss	%xmm0, -56(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L26
.L47:
	nop
.L26:
	jmp	.L38
.L43:
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	findValueAt
	.type	findValueAt, @function
findValueAt:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movq	$0, -8(%rbp)
.L51:
	cmpq	$0, -8(%rbp)
	jne	.L54
	movss	-20(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	movaps	%xmm0, %xmm1
	mulss	-20(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	movaps	%xmm0, %xmm2
	addss	%xmm0, %xmm2
	subss	%xmm2, %xmm1
	movaps	%xmm1, %xmm0
	movss	.LC9(%rip), %xmm1
	subss	%xmm1, %xmm0
	jmp	.L53
.L54:
	nop
	jmp	.L51
.L53:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	findValueAt, .-findValueAt
	.globl	bisect
	.type	bisect, @function
bisect:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	$0, -8(%rbp)
.L58:
	cmpq	$0, -8(%rbp)
	jne	.L61
	movss	-20(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	movss	.LC10(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L60
.L61:
	nop
	jmp	.L58
.L60:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	bisect, .-bisect
	.section	.rodata
	.align 4
.LC9:
	.long	1084227584
	.align 4
.LC10:
	.long	1073741824
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
