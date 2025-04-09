	.file	"Naz-updates_C-programs_addswap_flatten.c"
	.text
	.globl	_TIG_IZ_D1I9_envp
	.bss
	.align 8
	.type	_TIG_IZ_D1I9_envp, @object
	.size	_TIG_IZ_D1I9_envp, 8
_TIG_IZ_D1I9_envp:
	.zero	8
	.globl	_TIG_IZ_D1I9_argc
	.align 4
	.type	_TIG_IZ_D1I9_argc, @object
	.size	_TIG_IZ_D1I9_argc, 4
_TIG_IZ_D1I9_argc:
	.zero	4
	.globl	_TIG_IZ_D1I9_argv
	.align 8
	.type	_TIG_IZ_D1I9_argv, @object
	.size	_TIG_IZ_D1I9_argv, 8
_TIG_IZ_D1I9_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"after swapping a=%d b=%d\n"
	.text
	.globl	swap
	.type	swap, @function
swap:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$0, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L5
.L4:
	movq	$1, -8(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	swap, .-swap
	.section	.rodata
.LC1:
	.string	"sum = %d\n"
.LC2:
	.string	"enter two numbers"
.LC3:
	.string	"%d%d"
.LC4:
	.string	"1.add\n2.swap"
.LC5:
	.string	"enter your choice"
.LC6:
	.string	"%d"
.LC7:
	.string	"invalid choice"
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
	movq	$0, _TIG_IZ_D1I9_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_D1I9_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_D1I9_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-D1I9--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_D1I9_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_D1I9_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_D1I9_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L29:
	cmpq	$9, -16(%rbp)
	ja	.L32
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L23-.L17
	.long	.L32-.L17
	.long	.L33-.L17
	.long	.L21-.L17
	.long	.L32-.L17
	.long	.L20-.L17
	.long	.L19-.L17
	.long	.L32-.L17
	.long	.L18-.L17
	.long	.L16-.L17
	.text
.L18:
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L24
.L21:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-44(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, -32(%rbp)
	leaq	-44(%rbp), %rax
	movq	%rax, -24(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -16(%rbp)
	jmp	.L24
.L16:
	leaq	-44(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	swap
	movq	$2, -16(%rbp)
	jmp	.L24
.L19:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L24
.L20:
	movl	-40(%rbp), %eax
	cmpl	$1, %eax
	je	.L25
	cmpl	$2, %eax
	jne	.L26
	movq	$9, -16(%rbp)
	jmp	.L27
.L25:
	movq	$8, -16(%rbp)
	jmp	.L27
.L26:
	movq	$6, -16(%rbp)
	nop
.L27:
	jmp	.L24
.L23:
	movq	$3, -16(%rbp)
	jmp	.L24
.L32:
	nop
.L24:
	jmp	.L29
.L33:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
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
