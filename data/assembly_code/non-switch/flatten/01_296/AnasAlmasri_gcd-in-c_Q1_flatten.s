	.file	"AnasAlmasri_gcd-in-c_Q1_flatten.c"
	.text
	.globl	_TIG_IZ_4GGc_envp
	.bss
	.align 8
	.type	_TIG_IZ_4GGc_envp, @object
	.size	_TIG_IZ_4GGc_envp, 8
_TIG_IZ_4GGc_envp:
	.zero	8
	.globl	_TIG_IZ_4GGc_argv
	.align 8
	.type	_TIG_IZ_4GGc_argv, @object
	.size	_TIG_IZ_4GGc_argv, 8
_TIG_IZ_4GGc_argv:
	.zero	8
	.globl	_TIG_IZ_4GGc_argc
	.align 4
	.type	_TIG_IZ_4GGc_argc, @object
	.size	_TIG_IZ_4GGc_argc, 4
_TIG_IZ_4GGc_argc:
	.zero	4
	.text
	.globl	gcd_calculator
	.type	gcd_calculator, @function
gcd_calculator:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$1, -8(%rbp)
.L11:
	cmpq	$5, -8(%rbp)
	je	.L2
	cmpq	$5, -8(%rbp)
	ja	.L13
	cmpq	$3, -8(%rbp)
	je	.L4
	cmpq	$3, -8(%rbp)
	ja	.L13
	cmpq	$1, -8(%rbp)
	je	.L5
	cmpq	$2, -8(%rbp)
	je	.L6
	jmp	.L13
.L5:
	movq	$5, -8(%rbp)
	jmp	.L7
.L4:
	movl	-20(%rbp), %eax
	jmp	.L12
.L2:
	cmpl	$0, -24(%rbp)
	je	.L9
	movq	$2, -8(%rbp)
	jmp	.L7
.L9:
	movq	$3, -8(%rbp)
	jmp	.L7
.L6:
	movl	-20(%rbp), %eax
	cltd
	idivl	-24(%rbp)
	movl	%edx, -12(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L7
.L13:
	nop
.L7:
	jmp	.L11
.L12:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	gcd_calculator, .-gcd_calculator
	.section	.rodata
	.align 8
.LC0:
	.string	"(1) to restart program, (-1) to end program."
.LC1:
	.string	"%d"
.LC2:
	.string	"Invalid input."
.LC3:
	.string	"Greatest Common Divisor: %d\n\n"
.LC4:
	.string	"\nEnter two integers: "
.LC5:
	.string	"%d %d"
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
	movq	$0, _TIG_IZ_4GGc_envp(%rip)
	nop
.L15:
	movq	$0, _TIG_IZ_4GGc_argv(%rip)
	nop
.L16:
	movl	$0, _TIG_IZ_4GGc_argc(%rip)
	nop
	nop
.L17:
.L18:
#APP
# 130 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4GGc--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_4GGc_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_4GGc_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_4GGc_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L52:
	cmpq	$22, -16(%rbp)
	ja	.L55
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L33-.L21
	.long	.L55-.L21
	.long	.L32-.L21
	.long	.L55-.L21
	.long	.L31-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L55-.L21
	.long	.L55-.L21
	.long	.L28-.L21
	.long	.L55-.L21
	.long	.L27-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L55-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L24:
	movl	-40(%rbp), %eax
	movl	%eax, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L38
.L33:
	movl	-36(%rbp), %edx
	movl	-40(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L39
	movq	$0, -16(%rbp)
	jmp	.L38
.L39:
	movq	$13, -16(%rbp)
	jmp	.L38
.L27:
	movl	-36(%rbp), %eax
	testl	%eax, %eax
	jns	.L41
	movq	$22, -16(%rbp)
	jmp	.L38
.L41:
	movq	$4, -16(%rbp)
	jmp	.L38
.L31:
	movl	-40(%rbp), %eax
	testl	%eax, %eax
	jns	.L43
	movq	$20, -16(%rbp)
	jmp	.L38
.L43:
	movq	$15, -16(%rbp)
	jmp	.L38
.L36:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$16, -16(%rbp)
	jmp	.L38
.L34:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L53
	jmp	.L54
.L26:
	movl	-32(%rbp), %eax
	cmpl	$-1, %eax
	je	.L46
	movq	$10, -16(%rbp)
	jmp	.L38
.L46:
	movq	$3, -16(%rbp)
	jmp	.L38
.L22:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L38
.L30:
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L38
.L28:
	movl	-36(%rbp), %eax
	testl	%eax, %eax
	jne	.L48
	movq	$18, -16(%rbp)
	jmp	.L38
.L48:
	movq	$2, -16(%rbp)
	jmp	.L38
.L25:
	cmpl	$2, -24(%rbp)
	je	.L50
	movq	$21, -16(%rbp)
	jmp	.L38
.L50:
	movq	$8, -16(%rbp)
	jmp	.L38
.L32:
	movl	$1, -32(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L38
.L20:
	movl	-36(%rbp), %eax
	negl	%eax
	movl	%eax, -36(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L38
.L29:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L38
.L37:
	movl	-40(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -40(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L38
.L35:
	movl	-36(%rbp), %edx
	movl	-40(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	gcd_calculator
	movl	%eax, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L38
.L23:
	movl	-40(%rbp), %eax
	negl	%eax
	movl	%eax, -40(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L38
.L55:
	nop
.L38:
	jmp	.L52
.L54:
	call	__stack_chk_fail@PLT
.L53:
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
