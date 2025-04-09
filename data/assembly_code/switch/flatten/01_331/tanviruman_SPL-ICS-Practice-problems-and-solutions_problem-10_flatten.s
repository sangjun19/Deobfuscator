	.file	"tanviruman_SPL-ICS-Practice-problems-and-solutions_problem-10_flatten.c"
	.text
	.globl	_TIG_IZ_ihe5_argc
	.bss
	.align 4
	.type	_TIG_IZ_ihe5_argc, @object
	.size	_TIG_IZ_ihe5_argc, 4
_TIG_IZ_ihe5_argc:
	.zero	4
	.globl	_TIG_IZ_ihe5_envp
	.align 8
	.type	_TIG_IZ_ihe5_envp, @object
	.size	_TIG_IZ_ihe5_envp, 8
_TIG_IZ_ihe5_envp:
	.zero	8
	.globl	_TIG_IZ_ihe5_argv
	.align 8
	.type	_TIG_IZ_ihe5_argv, @object
	.size	_TIG_IZ_ihe5_argv, 8
_TIG_IZ_ihe5_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"DIVI: %lf\n"
	.align 8
.LC1:
	.string	"Enter an expression ( 7 + 3): "
.LC2:
	.string	"%lf %c %lf"
	.align 8
.LC3:
	.string	"Zero as divisor is not valid!."
.LC4:
	.string	"ADD: %lf\n"
.LC5:
	.string	"MULTI: %lf\n"
.LC6:
	.string	"Invalid operator."
.LC8:
	.string	"SUB: %lf\n"
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
	movq	$0, _TIG_IZ_ihe5_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ihe5_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ihe5_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ihe5--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_ihe5_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_ihe5_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_ihe5_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L30:
	cmpq	$14, -16(%rbp)
	ja	.L34
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L34-.L8
	.long	.L15-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L34-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movzbl	-33(%rbp), %eax
	movsbl	%al, %eax
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
	movq	$7, -16(%rbp)
	jmp	.L24
.L22:
	movq	$10, -16(%rbp)
	jmp	.L24
.L21:
	movq	$2, -16(%rbp)
	jmp	.L24
.L23:
	movq	$13, -16(%rbp)
	jmp	.L24
.L20:
	movq	$0, -16(%rbp)
	nop
.L24:
	jmp	.L25
.L7:
	movsd	-32(%rbp), %xmm0
	movsd	-24(%rbp), %xmm1
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rcx
	leaq	-33(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L25
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L33
.L10:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L12:
	movq	$8, -16(%rbp)
	jmp	.L25
.L9:
	movsd	-32(%rbp), %xmm1
	movsd	-24(%rbp), %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L11:
	movsd	-32(%rbp), %xmm1
	movsd	-24(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L18:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L14:
	movsd	-24(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	ucomisd	%xmm1, %xmm0
	jp	.L32
	pxor	%xmm1, %xmm1
	ucomisd	%xmm1, %xmm0
	je	.L27
.L32:
	movq	$14, -16(%rbp)
	jmp	.L25
.L27:
	movq	$11, -16(%rbp)
	jmp	.L25
.L16:
	movsd	-32(%rbp), %xmm0
	movsd	-24(%rbp), %xmm1
	subsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L25
.L34:
	nop
.L25:
	jmp	.L30
.L33:
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
