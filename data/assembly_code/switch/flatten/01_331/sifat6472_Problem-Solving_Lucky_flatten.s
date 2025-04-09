	.file	"sifat6472_Problem-Solving_Lucky_flatten.c"
	.text
	.globl	last3sum
	.bss
	.align 4
	.type	last3sum, @object
	.size	last3sum, 4
last3sum:
	.zero	4
	.globl	_TIG_IZ_5wR9_argc
	.align 4
	.type	_TIG_IZ_5wR9_argc, @object
	.size	_TIG_IZ_5wR9_argc, 4
_TIG_IZ_5wR9_argc:
	.zero	4
	.globl	idx
	.align 4
	.type	idx, @object
	.size	idx, 4
idx:
	.zero	4
	.globl	_TIG_IZ_5wR9_envp
	.align 8
	.type	_TIG_IZ_5wR9_envp, @object
	.size	_TIG_IZ_5wR9_envp, 8
_TIG_IZ_5wR9_envp:
	.zero	8
	.globl	ticket
	.type	ticket, @object
	.size	ticket, 7
ticket:
	.zero	7
	.globl	first3sum
	.align 4
	.type	first3sum, @object
	.size	first3sum, 4
first3sum:
	.zero	4
	.globl	_TIG_IZ_5wR9_argv
	.align 8
	.type	_TIG_IZ_5wR9_argv, @object
	.size	_TIG_IZ_5wR9_argv, 8
_TIG_IZ_5wR9_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
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
	movl	$0, last3sum(%rip)
	nop
.L2:
	movl	$0, first3sum(%rip)
	nop
.L3:
	movl	$0, idx(%rip)
	nop
.L4:
	movl	$0, -24(%rbp)
	jmp	.L5
.L6:
	movl	-24(%rbp), %eax
	cltq
	leaq	ticket(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -24(%rbp)
.L5:
	cmpl	$6, -24(%rbp)
	jle	.L6
	nop
.L7:
	movq	$0, _TIG_IZ_5wR9_envp(%rip)
	nop
.L8:
	movq	$0, _TIG_IZ_5wR9_argv(%rip)
	nop
.L9:
	movl	$0, _TIG_IZ_5wR9_argc(%rip)
	nop
	nop
.L10:
.L11:
#APP
# 137 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5wR9--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_5wR9_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_5wR9_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_5wR9_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L24:
	cmpq	$8, -16(%rbp)
	ja	.L27
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L14(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L14(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L14:
	.long	.L27-.L14
	.long	.L27-.L14
	.long	.L19-.L14
	.long	.L18-.L14
	.long	.L17-.L14
	.long	.L16-.L14
	.long	.L27-.L14
	.long	.L15-.L14
	.long	.L13-.L14
	.text
.L17:
	call	solve
	movq	$3, -16(%rbp)
	jmp	.L20
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L25
	jmp	.L26
.L18:
	movl	-28(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-28(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -28(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L20
.L16:
	cmpl	$0, -20(%rbp)
	je	.L22
	movq	$4, -16(%rbp)
	jmp	.L20
.L22:
	movq	$8, -16(%rbp)
	jmp	.L20
.L15:
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	getchar@PLT
	movq	$3, -16(%rbp)
	jmp	.L20
.L19:
	movq	$7, -16(%rbp)
	jmp	.L20
.L27:
	nop
.L20:
	jmp	.L24
.L26:
	call	__stack_chk_fail@PLT
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC1:
	.string	"NO"
.LC2:
	.string	"YES"
.LC3:
	.string	"%s"
	.text
	.globl	solve
	.type	solve, @function
solve:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$15, -8(%rbp)
.L53:
	cmpq	$16, -8(%rbp)
	ja	.L54
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L31(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L31(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L31:
	.long	.L42-.L31
	.long	.L41-.L31
	.long	.L40-.L31
	.long	.L39-.L31
	.long	.L38-.L31
	.long	.L55-.L31
	.long	.L54-.L31
	.long	.L54-.L31
	.long	.L36-.L31
	.long	.L35-.L31
	.long	.L54-.L31
	.long	.L54-.L31
	.long	.L54-.L31
	.long	.L34-.L31
	.long	.L33-.L31
	.long	.L32-.L31
	.long	.L30-.L31
	.text
.L38:
	movl	idx(%rip), %eax
	cltq
	leaq	ticket(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	leal	-48(%rax), %edx
	movl	first3sum(%rip), %eax
	addl	%edx, %eax
	movl	%eax, first3sum(%rip)
	movq	$2, -8(%rbp)
	jmp	.L43
.L33:
	movl	idx(%rip), %eax
	cltq
	leaq	ticket(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	testb	%al, %al
	je	.L44
	movq	$16, -8(%rbp)
	jmp	.L43
.L44:
	movq	$8, -8(%rbp)
	jmp	.L43
.L32:
	movq	$13, -8(%rbp)
	jmp	.L43
.L36:
	movl	first3sum(%rip), %edx
	movl	last3sum(%rip), %eax
	cmpl	%eax, %edx
	jne	.L46
	movq	$9, -8(%rbp)
	jmp	.L43
.L46:
	movq	$1, -8(%rbp)
	jmp	.L43
.L41:
	leaq	.LC1(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L43
.L39:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L43
.L30:
	movl	idx(%rip), %eax
	cmpl	$2, %eax
	jg	.L48
	testl	%eax, %eax
	jns	.L49
	jmp	.L50
.L48:
	subl	$3, %eax
	cmpl	$2, %eax
	ja	.L50
	movq	$0, -8(%rbp)
	jmp	.L51
.L49:
	movq	$4, -8(%rbp)
	jmp	.L51
.L50:
	movq	$2, -8(%rbp)
	nop
.L51:
	jmp	.L43
.L35:
	leaq	.LC2(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L43
.L34:
	movl	$0, first3sum(%rip)
	movl	$0, last3sum(%rip)
	leaq	ticket(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, idx(%rip)
	movq	$14, -8(%rbp)
	jmp	.L43
.L42:
	movl	idx(%rip), %eax
	cltq
	leaq	ticket(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	leal	-48(%rax), %edx
	movl	last3sum(%rip), %eax
	addl	%edx, %eax
	movl	%eax, last3sum(%rip)
	movq	$2, -8(%rbp)
	jmp	.L43
.L40:
	movl	idx(%rip), %eax
	addl	$1, %eax
	movl	%eax, idx(%rip)
	movq	$14, -8(%rbp)
	jmp	.L43
.L54:
	nop
.L43:
	jmp	.L53
.L55:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	solve, .-solve
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
