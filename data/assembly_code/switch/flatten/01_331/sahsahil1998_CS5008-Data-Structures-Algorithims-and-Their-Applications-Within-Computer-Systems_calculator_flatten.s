	.file	"sahsahil1998_CS5008-Data-Structures-Algorithims-and-Their-Applications-Within-Computer-Systems_calculator_flatten.c"
	.text
	.globl	_TIG_IZ_135d_argv
	.bss
	.align 8
	.type	_TIG_IZ_135d_argv, @object
	.size	_TIG_IZ_135d_argv, 8
_TIG_IZ_135d_argv:
	.zero	8
	.globl	_TIG_IZ_135d_envp
	.align 8
	.type	_TIG_IZ_135d_envp, @object
	.size	_TIG_IZ_135d_envp, 8
_TIG_IZ_135d_envp:
	.zero	8
	.globl	_TIG_IZ_135d_argc
	.align 4
	.type	_TIG_IZ_135d_argc, @object
	.size	_TIG_IZ_135d_argc, 4
_TIG_IZ_135d_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Addition"
.LC1:
	.string	"Sum= %.2lf\n"
.LC2:
	.string	"Division"
.LC3:
	.string	"Quotient= %.2lf\n"
.LC4:
	.string	"Multiplication"
.LC5:
	.string	"Product= %.2lf\n"
.LC6:
	.string	"Enter your choice"
	.align 8
.LC7:
	.string	" 1. Addition\n 2. Subtraction\n 3. Multiplication\n 4. Division"
.LC8:
	.string	"%d"
.LC9:
	.string	"Enter a and b values: "
.LC10:
	.string	"%lf %lf"
	.align 8
.LC11:
	.string	"Please select from list of choices"
.LC12:
	.string	"Subtraction"
.LC13:
	.string	"Difference= %.2lf\n"
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
	movq	$0, _TIG_IZ_135d_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_135d_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_135d_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-135d--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_135d_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_135d_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_135d_envp(%rip)
	nop
	movq	$14, -24(%rbp)
.L25:
	cmpq	$16, -24(%rbp)
	ja	.L28
	movq	-24(%rbp), %rax
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L28-.L8
	.long	.L12-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L28-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L13:
	movsd	-40(%rbp), %xmm1
	movsd	-32(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -24(%rbp)
	jmp	.L17
.L9:
	movq	$16, -24(%rbp)
	jmp	.L17
.L10:
	movsd	-40(%rbp), %xmm0
	movsd	-32(%rbp), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -24(%rbp)
	jmp	.L17
.L14:
	movsd	-40(%rbp), %xmm1
	movsd	-32(%rbp), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -24(%rbp)
	jmp	.L17
.L7:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$10, -24(%rbp)
	jmp	.L17
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L11:
	movl	-44(%rbp), %eax
	cmpl	$4, %eax
	je	.L19
	cmpl	$4, %eax
	jg	.L20
	cmpl	$3, %eax
	je	.L21
	cmpl	$3, %eax
	jg	.L20
	cmpl	$1, %eax
	je	.L22
	cmpl	$2, %eax
	je	.L23
	jmp	.L20
.L19:
	movq	$12, -24(%rbp)
	jmp	.L24
.L21:
	movq	$3, -24(%rbp)
	jmp	.L24
.L23:
	movq	$2, -24(%rbp)
	jmp	.L24
.L22:
	movq	$4, -24(%rbp)
	jmp	.L24
.L20:
	movq	$0, -24(%rbp)
	nop
.L24:
	jmp	.L17
.L16:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -24(%rbp)
	jmp	.L17
.L15:
	movsd	-40(%rbp), %xmm0
	movsd	-32(%rbp), %xmm1
	subsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -24(%rbp)
	jmp	.L17
.L28:
	nop
.L17:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
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
