	.file	"DominicWKelley_C-Programming-Exercises_firstUniqueChar_flatten.c"
	.text
	.globl	_TIG_IZ_NIPz_envp
	.bss
	.align 8
	.type	_TIG_IZ_NIPz_envp, @object
	.size	_TIG_IZ_NIPz_envp, 8
_TIG_IZ_NIPz_envp:
	.zero	8
	.globl	_TIG_IZ_NIPz_argv
	.align 8
	.type	_TIG_IZ_NIPz_argv, @object
	.size	_TIG_IZ_NIPz_argv, 8
_TIG_IZ_NIPz_argv:
	.zero	8
	.globl	_TIG_IZ_NIPz_argc
	.align 4
	.type	_TIG_IZ_NIPz_argc, @object
	.size	_TIG_IZ_NIPz_argc, 4
_TIG_IZ_NIPz_argc:
	.zero	4
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
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
	movq	$0, _TIG_IZ_NIPz_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_NIPz_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_NIPz_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NIPz--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_NIPz_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_NIPz_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_NIPz_envp(%rip)
	nop
	movq	$0, -56(%rbp)
.L11:
	cmpq	$2, -56(%rbp)
	je	.L6
	cmpq	$2, -56(%rbp)
	ja	.L14
	cmpq	$0, -56(%rbp)
	je	.L8
	cmpq	$1, -56(%rbp)
	jne	.L14
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L8:
	movq	$2, -56(%rbp)
	jmp	.L10
.L6:
	movb	$97, -48(%rbp)
	movb	$98, -47(%rbp)
	movb	$99, -46(%rbp)
	movb	$100, -45(%rbp)
	movb	$101, -44(%rbp)
	movb	$102, -43(%rbp)
	movb	$103, -42(%rbp)
	movb	$104, -41(%rbp)
	movb	$105, -40(%rbp)
	movb	$106, -39(%rbp)
	movb	$107, -38(%rbp)
	movb	$108, -37(%rbp)
	movb	$109, -36(%rbp)
	movb	$110, -35(%rbp)
	movb	$111, -34(%rbp)
	movb	$112, -33(%rbp)
	movb	$97, -32(%rbp)
	movb	$98, -31(%rbp)
	movb	$99, -30(%rbp)
	movb	$100, -29(%rbp)
	movb	$101, -28(%rbp)
	movb	$102, -27(%rbp)
	movb	$103, -26(%rbp)
	movb	$104, -25(%rbp)
	movb	$105, -24(%rbp)
	movb	$0, -23(%rbp)
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	print_first_unique
	movq	$1, -56(%rbp)
	jmp	.L10
.L14:
	nop
.L10:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC0:
	.string	"%c\n"
.LC1:
	.string	"No unique character found!"
	.text
	.globl	print_first_unique
	.type	print_first_unique, @function
print_first_unique:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$20, -16(%rbp)
.L47:
	cmpq	$20, -16(%rbp)
	ja	.L48
	movq	-16(%rbp), %rax
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
	.long	.L48-.L18
	.long	.L48-.L18
	.long	.L32-.L18
	.long	.L31-.L18
	.long	.L30-.L18
	.long	.L29-.L18
	.long	.L28-.L18
	.long	.L48-.L18
	.long	.L27-.L18
	.long	.L26-.L18
	.long	.L25-.L18
	.long	.L24-.L18
	.long	.L48-.L18
	.long	.L23-.L18
	.long	.L48-.L18
	.long	.L48-.L18
	.long	.L49-.L18
	.long	.L21-.L18
	.long	.L20-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L20:
	addl	$1, -20(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L33
.L30:
	movl	-24(%rbp), %eax
	cmpl	-20(%rbp), %eax
	je	.L34
	movq	$10, -16(%rbp)
	jmp	.L33
.L34:
	movq	$18, -16(%rbp)
	jmp	.L33
.L27:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, -28(%rbp)
	movb	$0, -30(%rbp)
	movb	$0, -29(%rbp)
	movl	$0, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L33
.L31:
	movl	-24(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L36
	movq	$11, -16(%rbp)
	jmp	.L33
.L36:
	movq	$2, -16(%rbp)
	jmp	.L33
.L24:
	movb	$0, -30(%rbp)
	movl	$0, -20(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L33
.L26:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movb	$1, -29(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L33
.L23:
	movl	-20(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L39
	movq	$5, -16(%rbp)
	jmp	.L33
.L39:
	movq	$19, -16(%rbp)
	jmp	.L33
.L19:
	movzbl	-30(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L41
	movq	$9, -16(%rbp)
	jmp	.L33
.L41:
	movq	$6, -16(%rbp)
	jmp	.L33
.L21:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -16(%rbp)
	jmp	.L33
.L28:
	addl	$1, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L33
.L29:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movl	-20(%rbp), %eax
	movslq	%eax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L43
	movq	$4, -16(%rbp)
	jmp	.L33
.L43:
	movq	$18, -16(%rbp)
	jmp	.L33
.L25:
	movb	$1, -30(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L33
.L32:
	movzbl	-29(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L45
	movq	$17, -16(%rbp)
	jmp	.L33
.L45:
	movq	$16, -16(%rbp)
	jmp	.L33
.L17:
	movq	$8, -16(%rbp)
	jmp	.L33
.L48:
	nop
.L33:
	jmp	.L47
.L49:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	print_first_unique, .-print_first_unique
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
